import time
import pickle
from rebar import recording
from rebar.arrdict import stack, numpyify
from rebar import paths, parallel, plots, recording
import logging
import threading
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

def length(d):
    if isinstance(d, dict):
        (l,) = set(length(v) for v in d.values())
        return l
    return d.shape[0]

def _array(plot, state):
    fig = plot(state)
    arr = plots.array(fig)
    plt.close(fig)
    return arr

def _encode(plot, states, fps):
    with parallel.parallel(_array, progress=False) as p, \
            recording.Encoder(fps) as encoder:

        futures = [p(plot, states[i]) for i in range(length(states))]
        for future in futures:
            while not future.done():
                yield
            encoder(future.result())

    return encoder.value

def parasite(run_name, env, env_idx=0, length=64, period=60):

    start = time.time()
    states = []
    path = paths.path(run_name, 'recording').with_suffix('.mp4')
    while True:
        if time.time() > start:
            states.append(env.state(env_idx))
            yield

        if len(states) == length:
            log.info('Starting encoding')
            states = numpyify(stack(states))
            content = yield from _encode(env._plot, states, env.options.fps)
            path.write_bytes(content)

            states = []
            start = start + period
        
def render(run_name=-1):
    path = paths.path(run_name, 'recording')
    if not path.exists():
        raise ValueError(f'No recording for "{run_name}"')
    pickle.loads(path.read_bytes())

