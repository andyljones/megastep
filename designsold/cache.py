import requests
from tqdm.auto import tqdm
from io import BytesIO
import time
import os
import gzip
import pickle
import inspect 
from pathlib import Path
from functools import wraps
import rebar
import logging

log = logging.getLogger(__name__)

def download(url):
    bs = BytesIO()
    log.info(f'Downloading {url}')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers['Content-Length']) if 'Content-Length' in r.headers else None
        with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit='B') as pbar:
            for chunk in r.iter_content(chunk_size=2**20): 
                pbar.update(len(chunk))
                bs.write(chunk)
    return bs.getvalue()

def _diskcache(path, f, *args, **kwargs):
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        path.write_bytes(gzip.compress(pickle.dumps(f(*args, **kwargs), protocol=4)))
    return pickle.loads(gzip.decompress(path.read_bytes()))

def _diskclear(path):
    if path.exists():
        path.unlink()

def _memcache(cache, path, f, *args, **kwargs):
    if path not in cache:
        cache[path] = f(*args, **kwargs)
    return cache[path]

def _memclear(cache, path):
    if path in cache:
        del cache[path]

def _timecache(duration, cache, path, f, *args, **kwargs):
    if path in cache:
        calltime, val = cache[path]
        if (time.time() - calltime) < duration:
            return val
    calltime, val = time.time(), f(*args, **kwargs)
    cache[path] = (calltime, val)
    return val

def _cachepather(f, filepattern, root):
    return cachepath

def autocache(filepattern=None, disk=True, memory=False, duration=None, root='.cache'):
    """Uses the modulename, function name and arguments to cache the results of a
    function in a sensible location. For example, suppose you have a function called 
    `transactions` in a module called `banks.starling`. It takes one argument, a date.
    Then by decorating it as
    ```
    @autocache('{date:%Y-%m-%d}')
    def transactions(date):
        ...
    ```
    when you call `transactions(pd.Timestamp('2018-11-22'))`, the result will be stored
    to `.cache/banks/starling/transactions/2018-11-22`. Next time you call it with the 
    same argument, the result will be loaded from that cache file.
    If you leave the pattern empty, it'll default to a concatenation of the params. So
    you could have easily written
    ```
    @autocache()
    def transactions(date):
        ...
    ```
    What's more, you can also set `memory=True` and get an additional in-memory cache 
    that wraps the disk cache. If a result is in memory, that'll be returned, else 
    it'll go to the disk, and only if the result is missing there too will the
    function be called.
    """  

    # Default to `.cache/slashed/module/path`
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0]).__name__

    def decorator(f):
        cache = {}

        nonlocal filepattern
        if filepattern is None:
            params = inspect.signature(f).parameters
            filepattern = '-'.join(f'{{{p}}}' for p in params)

        # If the function is parameterless, fall back to the base path
        parts = [root, *module.split('.'), f.__name__]
        parts = parts + [filepattern] if filepattern else parts
        pattern = os.path.join(*parts) 

        def cachepath(*args, **kwargs):
            bind = inspect.signature(f).bind(*args, **kwargs)
            bind.apply_defaults()
            return Path(pattern.format(**bind.arguments))
        
        @wraps(f)
        def wrapped(*args, **kwargs):
            path = cachepath(*args, **kwargs)
            if duration:
                #TODO: Implement disk duration caching.
                assert not disk, 'Can\'t specify a duration and use disk caching'
                return _timecache(duration, cache, path, f, *args, **kwargs)
            elif memory and disk:
                return _memcache(cache, path, _diskcache, path, f, *args, **kwargs)
            elif disk:
                return _diskcache(path, f, *args, **kwargs)
            elif memory:
                return _memcache(cache, path, f, *args, **kwargs)
            else:
                return f
    
        def clear(*args, **kwargs):
            path = cachepath(*args, **kwargs)
            if memory:
                _memclear(cache, path)
            if disk:
                _diskclear(path)
        
        wrapped.clear = clear
        return wrapped
    return decorator

def memcache(*args, **kwargs):
    return autocache(*args, **kwargs, memory=True, disk=False)

def timecache(duration, *args, **kwargs):
    return autocache(*args, **kwargs, duration=duration, disk=False)

