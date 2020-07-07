import time
import torch
from .. import numpy, paths, widgets, logging
import re
import numpy as np
from .. import arrdict
from . import categories
import pandas as pd
import threading
from contextlib import contextmanager
import _thread

log = logging.getLogger(__name__)

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
        return arrdict.arrdict(self._arrs)

    def pandas(self):
        arrs = self.arrays()

        dfs = {}
        for (category, field), arr in arrs.items():
            df = pd.DataFrame.from_records(arr, index='_time')
            df.index.name = 'time'
            dfs[category, field] = df
        return arrdict.arrdict(dfs)
        
    def resample(self, rule='60s', **kwargs):
        kwargs = {'rule': rule, **kwargs}

        results = {}
        for (category, field), df in self.pandas().items():
            func = getattr(categories, category)
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

def __from_dir(canceller, run_name, out, rule, throttle=1):
    reader = Reader(run_name)
    start = pd.Timestamp.now()

    nxt = time.time()
    while True:
        if time.time() > nxt:
            nxt = nxt + throttle

            # Base slightly into the future, else by the time the resample actually happens you're 
            # left with an almost-empty last interval.
            base = int(time.time() % 60) + 5
            values = reader.resample(rule=rule, base=base)
            
            if len(values) > 0:
                values = values.ffill(limit=1).iloc[-1].to_dict()
                key_length = max([len(str(k)) for k in values], default=0)+1
                content = '\n'.join(f'{{:{key_length}s}} {{}}'.format(k, format(values[k])) for k in sorted(values))
            else:
                content = 'No stats yet'

            size = paths.size(run_name, 'stats')
            age = pd.Timestamp.now() - start
            out.refresh(f'{run_name}: {tdformat(age)} old, {rule} rule, {size:.0f}MB on disk\n\n{content}')

        if canceller.is_set():
            break

        time.sleep(.1)

def _from_dir(*args, **kwargs):
    try:
        __from_dir(*args, **kwargs)
    except KeyboardInterrupt:
        log.info('Interrupting main')
        _thread.interrupt_main()

@contextmanager
def from_dir(run_name, compositor=None, rule='60s'):
    if logging.in_ipython():
        try:
            canceller = threading.Event()
            out = (compositor or widgets.Compositor()).output()
            thread = threading.Thread(target=_from_dir, args=(canceller, run_name, out, rule))
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