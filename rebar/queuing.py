import time
import torch
from torch import multiprocessing as mp
import queue
from contextlib import contextmanager, asynccontextmanager
import traceback
import asyncio
from functools import wraps
from torch import nn
import logging
from . import dotdict

log = logging.getLogger(__name__)

class SerialQueue:

    def __init__(self):
        self._queue = []
        self._put_end = False
        self._got_end = False

    def get(self):
        if len(self._queue) > 0:
            item = self._queue[0]
            self._queue = self._queue[1:]

            if isinstance(item, str) and (item == '__END__'):
                log.info('Got END')
                self._got_end = True
                return None
            else:
                return item
        else:
            return None
    
    def put(self, item):
        # Not safe to test `in` directly since `item` is likely to be a tensor
        if isinstance(item, (str, type(None))) and (item in ('__END__', None)):
            raise ValueError(f'Tried to put sentinel value "{item}"')
        if len(self._queue) < 1:
            self._queue.append(item)
            return True
        else:
            return False

    def put_end(self):
        if self._put_end:
            return True
        else:
            if len(self._queue) < 1:
                self._queue.append('__END__')
                log.info('Put END')
                self._put_end = True
                return True
            else:
                return False

    def get_end(self):
        self.get()
        return self._got_end
        
    def join(self, timeout=None):
        if len(self._queue) == 0:
            return True
        else:
            return False

class MultiprocessQueue:

    def __init__(self):
        self.queue = mp.JoinableQueue(1)
        self._put_end = False
        self._got_end = False

    def get(self):
        try:
            item = self.queue.get_nowait()
            if isinstance(item, str) and (item == '__END__'):
                log.info('Got END')
                self._got_end = True
                self.queue.task_done()
                return None
            else:
                self.queue.task_done()
                return item
        except queue.Empty:
            return None
    
    def put(self, item):
        # Not safe to test `in` directly since `item` is likely to be a tensor
        if isinstance(item, (str, type(None))) and (item in ('__END__', None)):
            raise ValueError(f'Tried to put sentinel value "{item}"')
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def put_end(self):
        try:
            if not self._put_end:
                self.queue.put_nowait('__END__')
                log.info('Put END')
                self._put_end = True
            return True
        except queue.Full:
            return False

    def get_end(self):
        self.get()
        return self._got_end
        
    def join(self, timeout=None):
        try:
            with self.queue._cond:
                if not self.queue._unfinished_tasks._semlock._is_zero():
                    self.queue._cond.wait(timeout=timeout)
            return True
        except RuntimeError:
            return False

async def close(intakes, outputs, timeout=5):
    """Strategy:
        * Wait until you can send an END through each output queue
        * Drain the intake queues until you get an END from each one
        * Wait for each output queue to drain
    """
    log.info(f'Closing; draining intakes and waiting to send ENDs. {timeout}s timeout.')
    cutoff = time.time() + timeout
    while True:
        # Avoid a deadlock where everyone's queues are full so ENDs can't be sent
        for intake in intakes:
            intake.get()
        
        if all(o.put_end() for o in outputs):
            break
        if time.time() > cutoff:
            log.warn('Timed out while waiting to send ENDs')
            return 

        # We're not actually running in a proper scheduler here, so can't sleep via it
        await asyncio.sleep(0)
        time.sleep(.1)
    
    log.info(f'Sent ENDs to outputs; waiting to get ENDs from intakes')
    while True:
        if all(i.get_end() for i in intakes):
            break
        if time.time() > cutoff:
            log.warn('Timed out while waiting to get ENDs')
            return

        # We're not actually running in a proper scheduler here, so can't sleep via it
        await asyncio.sleep(0)
        time.sleep(.1)

    log.info(f'Intakes emptied; waiting for outputs to drain')
    while True:
        if all(o.join(.1) for o in outputs):
            break
        if time.time() > cutoff:
            log.warn('Timed out while waiting to drain outputs')
            return

        # We're not actually running in a proper scheduler here, so can't sleep via it
        await asyncio.sleep(0)
        time.sleep(.1)

    log.info('Outputs drained.')

def create(x, serial=False):
    if isinstance(x, dict):
        return dotdict.dotdict({n: create(v, serial) for n, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return dotdict.dotdict({n: create(n, serial) for n in x})
    elif isinstance(x, str):
        return SerialQueue() if serial else MultiprocessQueue()
    raise ValueError(f'Can\'t handle {type(x)}')

@asynccontextmanager
async def cleanup(intakes, outputs):
    intakes = [intakes] if isinstance(intakes, (SerialQueue, MultiprocessQueue)) else intakes
    outputs = [outputs] if isinstance(outputs, (SerialQueue, MultiprocessQueue)) else outputs
    try:
        yield
    except:
        log.info(f'Got an exception, cleaning up queues:\n{traceback.format_exc()}')
        raise
    finally:
        await close(intakes, outputs)

