import torch
import numpy as np
import inspect
from ..contextlib import maybeasynccontextmanager
from .. import numpy
from . import categories
from functools import partial

__all__ = ['to_dir', 'defer', 'record']

WRITER = None

@maybeasynccontextmanager
def to_dir(run_name):
    try:
        global WRITER
        old = WRITER
        WRITER = numpy.Writer(run_name, 'stats')
        yield
    finally:
        WRITER = old
def clean(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x

def eager_record(category, field, *args, **kwargs):
    if WRITER is None:
        return 
    if not isinstance(field, str):
        raise ValueError(f'Field should be a string, is actually {field}')

    args = tuple(clean(a) for a in args)
    kwargs = {k: clean(v) for k, v in kwargs.items()}

    func = categories.CATEGORIES[category]
    call = inspect.getcallargs(func, *args, **kwargs)
    call = {'_time': np.datetime64('now'), **call}

    WRITER.write(f'{category}/{field}', call)

_record = eager_record
QUEUE = None

def record(*args, **kwargs):
    return _record(*args, **kwargs)
def deferred_record(category, field, *args, **kwargs):
    if not isinstance(field, str):
        raise ValueError(f'Field should be a string, is actually {field}')
    QUEUE.append((category, field, args, kwargs))

def _mono_getter(collection, x):
    dtype = x.dtype
    if dtype not in collection:
        collection[dtype] = []
    start = sum(c.nelement() for c in collection[dtype])
    end = start + x.nelement()
    collection[dtype].append(x.flatten())

    def f(collection):
        return collection[dtype][start:end].reshape(x.shape)
    return f

def _dummy_getter(x):
    def f(collection):
        return x
    return f

def _multi_getter(collection, *args, **kwargs):
    arggetters = []
    for a in args:
        if isinstance(a, torch.Tensor) and a.device.type != 'cpu':
            arggetters.append(_mono_getter(collection, a))
        else:
            arggetters.append(_dummy_getter(a))

    kwarggetters = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor) and v.device.type != 'cpu':
            kwarggetters[k] = _mono_getter(collection, v)
        else:
            kwarggetters[k] = _dummy_getter(v)

    def f(collection):
        args = tuple(g(collection) for g in arggetters)
        kwargs = {k: g(collection) for k, g in kwarggetters.items()}
        return args, kwargs
    return f

def _gather(queue):
    collection = {}
    getters = []
    for category, field, args, kwargs in queue:
        getters.append((category, field, _multi_getter(collection, *args, **kwargs)))
    collection = {k: torch.cat(v).detach().cpu() for k, v in collection.items()}
    return collection, getters

@maybeasynccontextmanager
def defer():
    global _record
    global QUEUE
    _record = deferred_record
    QUEUE = []
    try:
        yield
    finally:
        collection, getters = _gather(QUEUE)

        for (category, field, getter) in getters:
            args, kwargs = getter(collection)
            args = tuple(clean(a) for a in args)
            kwargs = {k: clean(v) for k, v in kwargs.items()}
            func = categories.CATEGORIES[category]
            call = inspect.getcallargs(func, *args, **kwargs)
            call = {'_time': np.datetime64('now'), **call}

            if WRITER is not None:
                WRITER.write(f'{category}/{field}', call)
    
        QUEUE = None
        _record = eager_record

for c in categories.CATEGORIES:
    locals()[c] = partial(record, c)
    __all__.append(c)