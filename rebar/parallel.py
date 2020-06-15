from tqdm.auto import tqdm
from contextlib import contextmanager
import multiprocessing
import types
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, _base, as_completed
import logging

log = logging.getLogger(__name__)

class SerialExecutor(_base.Executor):
    """An executor that runs things on the main process/thread - meaning stack traces are interpretable
    and the debugger works!
    """
    
    def __init__(self, *args, **kwargs):
        pass
    
    def submit(self, f, *args, **kwargs):
        future = Future()
        future.set_result(f(*args, **kwargs))
        return future

@contextmanager
def VariableExecutor(N=None, processes=True, **kwargs):
    """An executor that can be easily switched between serial, thread and parallel execution.
    If N=0, a serial executor will be used.
    """
    
    N = multiprocessing.cpu_count() if N is None else N
    
    if N == 0:
        executor = SerialExecutor
    elif processes:
        executor = ProcessPoolExecutor
    else:
        executor = ThreadPoolExecutor
    
    log.debug('Launching a {} with {} processes'.format(executor.__name__, N))    
    with executor(N, **kwargs) as pool:
        yield pool
        
@contextmanager
def parallel(f, progress=True, **kwargs):
    """Sugar for using the VariableExecutor. Call as
    
    with parallel(f) as g:
        ys = g.wait({x: g(x) for x in xs})
    and f'll be called in parallel on each x, and the results collected in a dictionary.
    A fantastic additonal feature is that if you pass `parallel(f, N=0)` , everything will be run on 
    the host process, so you can `import pdb; pdb.pm()` any errors. 
    """

    with VariableExecutor(**kwargs) as pool:

        def reraise(f, futures={}):
            e = f.exception()
            if e:
                log.warn('Exception raised on "{}"'.format(futures[f]), exc_info=e)
                raise e
            return f.result()

        submitted = set()

        def submit(*args, **kwargs):
            fut = pool.submit(f, *args, **kwargs)
            submitted.add(fut)
            fut.add_done_callback(submitted.discard)  # Try to avoid memory leak
            return fut
        
        def wait(c):
            # Recurse on list-likes
            if type(c) in (list, tuple, types.GeneratorType):
                ctor = list if isinstance(c, types.GeneratorType) else type(c)
                results = wait(dict(enumerate(c)))
                return ctor(results[k] for k in sorted(results))

            # Now can be sure we've got a dict-like
            futures = {fut: k for k, fut in c.items()}
            
            results = {}
            for fut in tqdm(as_completed(futures), total=len(c), disable=not progress):
                results[futures[fut]] = reraise(fut, futures)
                
            return results
        
        def cancel():
            while True:
                remaining = list(submitted)
                for fut in remaining:
                    fut.cancel()
                    submitted.discard(fut)
                if not remaining:
                    break

        try:
            submit.wait = wait
            yield submit
        finally:
            cancel()