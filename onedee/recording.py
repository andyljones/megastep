import time
import pickle
from rebar import recording
from rebar.arrdict import stack, numpyify
from rebar import paths, parallel, plots, recording
import logging
import threading
import matplotlib.pyplot as plt
import os
from functools import wraps

log = logging.getLogger(__name__)

def length(d):
    if isinstance(d, dict):
        (l,) = set(length(v) for v in d.values())
        return l
    return d.shape[0]

def _init():
    # Suppress keyboard interrupt of workers, since exiting the context 
    # manager in the parent will shut them down.
    import signal
    signal.signal(signal.SIGINT, lambda h, f: None)

def _array(plot, state):
    fig = plot(state)
    arr = plots.array(fig)
    plt.close(fig)
    return arr

def _encode(tasker, plot, states, fps):
    log.info('Started encoding recording')
    states = numpyify(stack(states))
    futures = [tasker(plot, states[i]) for i in range(length(states))]
    with recording.Encoder(fps) as encoder:
        for future in futures:
            while not future.done():
                yield
            encoder(future.result())
    log.info('Finished encoding recording')
    return encoder.value

def callable(f):

    @wraps(f)
    def g(*args, **kwargs):
        co = f(*args, **kwargs)
        co.send(None)
        step = lambda x: co.send(x)
        return step
    
    return g

@callable
def recorder(run_name, plot, env_idx=0, length=256, period=60, fps=20):
    start = time.time()
    states = []
    path = paths.path(run_name, 'recording').with_suffix('.mp4')
    with parallel.parallel(_array, progress=False, N=os.cpu_count()//4, initializer=_init) as tasker:
        while True:
            if time.time() > start:
                state = yield
                states.append(state)

            if len(states) == length:
                video = yield from _encode(tasker, plot, states, fps)
                path.write_bytes(video)

                states = []
                start = start + period
        
def notebook(run_name=-1, idx=-1):
    path = list(paths.subdirectory('test', 'recording').glob('*.mp4'))[idx]
    return recording.notebook(path.read_bytes())

