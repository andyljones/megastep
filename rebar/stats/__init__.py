import torch
import time
from contextlib import contextmanager
from functools import partial
from torch import nn
from . import categories
import logging

log = logging.getLogger(__name__)

# For re-export
from .writing import to_dir, defer, record
from .reading import from_dir, Reader

for c in categories.CATEGORIES:
    locals()[c] = partial(record, c)
# Defned to be used by other functions in this module
mean = partial(record, 'mean')

@contextmanager
def via_dir(run_name, compositor=None):
    with to_dir(run_name), from_dir(run_name, compositor):
        yield

def gpu_memory(name):
    total_mem = torch.cuda.get_device_properties('cuda').total_memory
    max(f'gpu-cache/{name}', torch.cuda.max_memory_cached()/total_mem)
    torch.cuda.reset_max_memory_cached()
    max(f'gpu-alloc/{name}', torch.cuda.max_memory_allocated()/total_mem)
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

def normhook(name, t):

    def hook(grad):
        mean(name, grad.pow(2).sum().pow(.5))

    t.register_hook(hook)

def total_gradient_norm(params):
    if isinstance(params, nn.Module):
        return total_gradient_norm(params.parameters())
    norms = [p.grad.data.float().pow(2).sum() for p in params if p.grad is not None]
    return torch.sum(torch.tensor(norms)).pow(.5)

def total_norm(params):
    if isinstance(params, nn.Module):
        return total_norm(params.parameters())
    return sum([p.data.float().pow(2).sum() for p in params if p is not None]).pow(.5)

def rel_gradient_norm(name, agent):
    mean(name, total_gradient_norm(agent), total_norm(agent))

def funcduty(name):
    def factory(f):
        def g(self, *args, **kwargs):
            start = time.time()
            result = f(self, *args, **kwargs)
            record('duty', f'duty/{name}', time.time() - start)
            return result
        return g
    return factory

## TESTS

def test_from_dir():
    from .. import paths, widgets, logging
    paths.clear('test-run', 'stats')
    paths.clear('test-run', 'logs')

    compositor = widgets.Compositor()
    with logging.from_dir('test-run', compositor), \
            to_dir('test-run'), \
            from_dir('test-run', compositor):
        for i in range(10):
            mean('count', i)
            mean('twocount', 2*i)
            time.sleep(.25) 
