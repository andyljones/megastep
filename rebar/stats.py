import torch
import numpy as np
import time
from . import widgets, logging, statscategories, numpy, paths
from .contextlib import maybeasynccontextmanager
from contextlib import contextmanager
import threading
import inspect
from functools import partial
import pandas as pd
from collections import defaultdict
import numpy as np
import urllib
import re
import _thread
from torch import nn
from .arrdict import arrdict

# Will be shadowed by the functions pulled in from statscategories
_max = max
_time = time

log = logging.getLogger(__name__)

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
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x

def record(category, field, *args, **kwargs):
    if WRITER is None:
        raise IOError(f'No writer set while trying to record a "{category}" called "{field}"')
    if not isinstance(field, str):
        raise ValueError(f'Field should be a string, is actually {field}')

    args = tuple(clean(a) for a in args)
    kwargs = {k: clean(v) for k, v in kwargs.items()}

    func = statscategories.CATEGORIES[category]
    call = inspect.getcallargs(func, *args, **kwargs)
    call = {'_time': np.datetime64('now'), **call}

    WRITER.write(f'{category}/{field}', call)

def format(v):
    if isinstance(v, int):
        return f'{v}'
    if isinstance(v, float):
        return f'{v:.6g}'
    if isinstance(v, list):
        return ', '.join(format(vv) for vv in v)
    if isinstance(v, dict):
        return '{' + ', '.join(f'{k}: {format(vv)}' for k, vv in v.items()) + '}'
    return str(v)

def adaptive_rule(df):
    timespan = (df.index[-1] - df.index[0]).total_seconds()
    if timespan < 600:
        return '15s'
    elif timespan < 7200:
        return '1min'
    else:
        return '10min'
    

class Reader:

    def __init__(self, run_name, prefix=''):
        self._reader = numpy.Reader(run_name, 'stats')
        self._prefix = prefix
        self._arrs = {}

    def arrays(self):
        #TODO: If this gets slow, do amortized allocation of arrays x2 as big as needed
        for channel, new in self._reader.read().items():
            category, field = re.match(r'^(.*?)/(.*)$', channel).groups()
            if field.startswith(self._prefix):
                current = [self._arrs[category, field]] if (category, field) in self._arrs else []
                self._arrs[category, field] = np.concatenate(current + new)
        return arrdict(self._arrs)

    def pandas(self):
        arrs = self.arrays()

        dfs = {}
        for (category, field), arr in arrs.items():
            df = pd.DataFrame.from_records(arr, index='_time')
            df.index.name = 'time'
            dfs[category, field] = df
        return arrdict(dfs)
        
    def resample(self, rule='60s', **kwargs):
        kwargs = {'rule': rule, **kwargs}

        results = {}
        for (category, field), df in self.pandas().items():
            func = getattr(statscategories, category)
            results[field] = func(**{k: df[k] for k in df})(**kwargs)

        if results:
            df = pd.concat(results, 1)
            df.index = df.index - df.index[0]
            return df
        else:
            return pd.DataFrame(index=pd.TimedeltaIndex([], name='time'))

def arrays(prefix='', run_name=-1):
    return Reader(run_name, prefix).arrays()

def pandas(name, run_name=-1):
    dfs = Reader(run_name, name).pandas()
    for (_, field), df in dfs.items():
        return df
    raise KeyError(f'Couldn\'t find a statistic matching {name}')

def resample(prefix='', run_name=-1, rule='60s'):
    return Reader(run_name, prefix).resample(rule)

def tdformat(td):
    """How is this not in Python, numpy or pandas?"""
    x = td.total_seconds()
    x, _ = divmod(x, 1)
    x, s = divmod(x, 60)
    if x < 1:
        return f'{s:.0f}s'
    h, m = divmod(x, 60)
    if h < 1:
        return f'{m:.0f}m{s:02.0f}s'
    else:
        return f'{h:.0f}h{m:02.0f}m{s:02.0f}s'

def __from_dir(canceller, run_name, out, throttle=1):
    reader = Reader(run_name)
    start = pd.Timestamp.now()

    nxt = 0
    while True:
        if _time.time() > nxt:
            nxt = nxt + throttle

            # Base slightly into the future, else by the time the resample actually happens you're 
            # left with an almost-empty last interval.
            base = int(time.time() % 60) + 5
            values = reader.resample(rule='60s', base=base)
            
            if len(values) > 0:
                values = values.ffill(limit=1).iloc[-1].to_dict()
                key_length = _max([len(str(k)) for k in values], default=0)+1
                content = '\n'.join(f'{{:{key_length}s}} {{}}'.format(k, format(values[k])) for k in sorted(values))
            else:
                content = 'No stats yet'

            size = paths.size(run_name, 'stats')
            age = pd.Timestamp.now() - start
            out.refresh(f'{run_name}: {tdformat(age)} old, {size:.0f}MB on disk\n\n{content}')

        if canceller.is_set():
            break

        _time.sleep(.001)

def _from_dir(canceller, run_name, out):
    try:
        __from_dir(canceller, run_name, out)
    except KeyboardInterrupt:
        log.info('Interrupting main')
        _thread.interrupt_main()

@contextmanager
def from_dir(run_name, compositor=None):
    if logging.in_ipython():
        try:
            canceller = threading.Event()
            out = (compositor or widgets.compositor()).output()
            thread = threading.Thread(target=_from_dir, args=(canceller, run_name, out))
            thread.start()
            yield
        finally:
            canceller.set()
            thread.join(1)
            if thread.is_alive():
                log.error('Stat display thread won\'t die')
            else:
                log.info('Stat display thread cancelled')

            # Want to leave the outputs open so you can see the final stats
            # out.close()
    else:
        log.info('No stats emitted in console mode')
        yield

@contextmanager
def via_dir(run_name, compositor=None):
    with to_dir(run_name), from_dir(run_name, compositor):
        yield

for c in statscategories.CATEGORIES:
    locals()[c] = partial(record, c)
# For other functions in this module
mean = partial(record, 'mean')

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
