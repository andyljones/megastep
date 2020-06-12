from contextlib import contextmanager, asynccontextmanager
from functools import wraps

class MaybeAsyncGeneratorContextManager:

    def __init__(self, func, args, kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._sync = None
        self._async = None

    def __enter__(self):
        if self._sync is None:
            syncfunc = contextmanager(self._func)
            self._sync = syncfunc(*self._args, **self._kwargs)
        return self._sync.__enter__()

    def __exit__(self, t, v, tb):
        return self._sync.__exit__(t, v, tb)

    def __aenter__(self):
        if self._async is None:
            @asynccontextmanager
            async def asyncfunc(*args, **kwargs):
                with contextmanager(self._func)(*args, **kwargs):
                    yield 
            self._async = asyncfunc(*self._args, **self._kwargs)
        return self._async.__aenter__()

    def __aexit__(self, t, v, tb):
        return self._async.__aexit__(t, v, tb)

def maybeasynccontextmanager(func):
    @wraps(func)
    def helper(*args, **kwds):
        return MaybeAsyncGeneratorContextManager(func, args, kwds)
    return helper