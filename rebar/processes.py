from torch import distributed as dist
import torch
import os
import signal
import asyncio
from functools import wraps
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import contextmanager
from torch.nn.parallel.distributed import _find_tensors
from .contextlib import maybeasynccontextmanager
import logging
import inspect
import time

log = logging.getLogger(__name__)

def initialize(device, devices):
    if dist.is_initialized():
        log.info('Process group is already initialized')
        return 

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    if device is None:
        os.environ['MASTER_PORT'] = str(29500)
        dist.init_process_group('nccl', rank=0, world_size=1)
    else:
        os.environ['MASTER_PORT'] = str(29500 + devices[0])
        dist.init_process_group('nccl', rank=devices.index(device), world_size=len(devices))

@contextmanager
def processgroup(device, devices):
    try:
        initialize(device, devices)
        yield
    finally:
        dist.destroy_process_group()

class DDP2(DDP):

    def forward(self, *inputs, **kwargs):
        """Modified to support not scattering inputs when it's only on one device"""
        if self.require_forward_param_sync:
            self._sync_params()

        if self.device_ids:
            if len(self.device_ids) == 1:
                output = self.module(*inputs, **kwargs)
            else:
                inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

        return output

def set_start_method():
    """If you use get_start_method to check what the start method is, you'll accidentally _set_ the start method
    and then get an error if you later try to set it. Here we open the box without killing the cat"""
    import os

    # https://github.com/pytorch/pytorch/issues/32575
    # os.environ['NCCL_BLOCKING_WAIT'] = '1'

    from multiprocessing import context
    ctx = context._default_context
    if ctx._actual_context is None:
        mp.set_start_method('spawn')
    else:
        assert ctx._actual_context._name in ('spawn', 'forkserver')

def consensus(b):
    b = torch.tensor(float(b)).cuda()
    dist.all_reduce(b, dist.ReduceOp.PRODUCT)
    return bool(b.cpu())

def cancel(canceller):
    if dist.is_initialized():
        # If we're in a process group, either the whole group needs to 
        # break or no-one does, else a process will be left hanging.
        cancel = canceller.is_set()
        if cancel:
            log.info('Canceller set, trying to break')
        if consensus(cancel):
            log.info('Everyone has cancelled, breaking')
            return True
    else:
        if canceller.is_set():
            log.info('Cancelled, breaking')
            return True

async def surrender():
    await asyncio.sleep(0)


class DeadStrand(Exception):
    pass

def coroutine_runner(f, *args, **kwargs):
    co = f(*args, **kwargs)
    try:
        while True:
            co.send(None)
    except StopIteration:
        pass
    except Exception as e:
        raise e


class ProcessSentinel:

    def __init__(self, wait=15):
        self._wait = wait
        self._processes = {}
        self._references = []
        self.canceller = mp.Event()
        set_start_method()

        self.serial = False

    def pin(self, obj):
        """There are sometimes objects passed to children - like queues - that need to 
        stick around for as long as the children do"""
        self._references.append(obj)

    def launch(self, f, *args, **kwargs):
        if (self.canceller not in args) and (self.canceller not in kwargs.values()):
            log.warn('Sentinel\'s canceller has not been passed to a launched process')

        count = len([n for n in self._processes if n == f.__qualname__])
        if inspect.iscoroutinefunction(f):
            proc = mp.Process(
                name=f'{f.__qualname__}-{count}',
                target=coroutine_runner,
                args=(f, *args),
                kwargs=kwargs)
        else:
            proc = mp.Process(
                name=f'{f.__qualname__}-{count}',
                target=f,
                args=args,
                kwargs=kwargs)
        proc.start()
        self._processes[f.__qualname__, count] = proc
        log.info(f'Launched process {f.__qualname__}-{count}')

    def wait(self):
        for _ in range(int(self._wait)):
            alive = [(n, c) for (n, c), p in self._processes.items() if p.is_alive()]
            if alive:
                strs = [f'{n}-{c}' for n, c in alive]
                log.info(f'Waiting for cancellations: {", ".join(strs)} still alive')
            else:
                log.info('All processes gracefully cancelled')
                break
            time.sleep(1)
        else:
            for n, c in alive:
                log.info(f'Failed to cancel "{n}-{c}"; terminating')
                self._processes[n].terminate() 

        self._references = []

    def cancel(self):
        log.info('Setting canceller')
        self.canceller.set()
        self.wait()

    def check(self):
        for (n, c), p in self._processes.items():
            if not p.is_alive():
                log.info(f'Process "{n}-{c}" died unexpectedly; cancelling')
                self.cancel()
                raise DeadStrand(f'Process "{n}-{c}" died unexpectedly')

class SerialSentinel:

    def __init__(self, wait=15):
        self._wait = wait
        self.canceller = mp.Event()

        self._coroutines = {}
        self._exited = []

        self.serial = True

    def launch(self, f, *args, **kwargs):
        if (self.canceller not in args) and (self.canceller not in kwargs.values()):
            log.warn('Sentinel\'s canceller has not been passed to a launched process')

        count = len([n for n, _ in self._coroutines if n == f.__qualname__])

        co = f(*args, **kwargs)
        self._coroutines[f.__qualname__, count] = co
        log.info(f'Launched coroutine {f.__qualname__}-{count}')

    def wait(self):
        for _ in range(int(self._wait)):
            alive = []
            for (n, c), co in self._coroutines.items():
                try:
                    co.send(None)
                except (RuntimeError, StopIteration):
                    pass
                else:
                    alive.append((n, c))
            if alive:
                strs = [f'{n}-{c}' for n, c in alive]
                log.info(f'Waiting for cancellations: {", ".join(strs)} still alive')
            else:
                log.info('All coroutines gracefully cancelled')
                break
        else:
            for n, c in alive:
                log.info(f'Failed to cancel "{n}-{c}"; closing')
                try:
                    self._coroutines[n, c].close()
                except RuntimeError:
                    pass

    def cancel(self):
        log.info('Setting canceller')
        self.canceller.set()
        self.wait()

    def check(self):
        for (n, c), co in self._coroutines.items():
            try:
                co.send(None)
            except StopIteration:
                pass
            except Exception as e:
                log.info(f'Coroutine "{n}-{c}" died unexpectedly; cancelling')
                self.cancel()
                raise e

@contextmanager
def sentinel(serial=False):
    sentinel = SerialSentinel() if serial else ProcessSentinel()
    try:
        yield sentinel
    except KeyboardInterrupt:
        log.info('Got a keyboard interrupt, cancelling processes')
        sentinel.cancel()
    except (DeadStrand,):
        raise
    except:
        sentinel.cancel()  
        raise
    else:
        sentinel.cancel()
