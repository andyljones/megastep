import time
import pickle
from rebar import recording
from rebar.arrdict import stack, numpyify
from rebar import paths, parallel, plots, recording
import logging
import threading

log = logging.getLogger(__name__)

def length(d):
    if isinstance(d, dict):
        (l,) = set(length(v) for v in d.values())
        return l
    return l.shape[0]

def _array(plot, state):
    return plots.array(plot(state))

def _encode(plot, states):
    with parallel.parallel(_array, progress=False) as p, \
            recording.Encoder() as encoder:

        futures = [p(states[i]) for i in range(length(states))]
        for future in futures:
            while not future.done():
                yield
            encoder(future.result())

        return encoder.value

def parasite(run_name, env, env_idx=0, length=512, period=60):

    start = time.time()
    states = []
    path = paths.path(run_name, 'recording').with_ext('.mp4')
    while True:
        if time.time() < start:
            yield

        states.append(env.state(env_idx))
        if len(states) == length:
            content = yield from _encode(plot, states)
            path.write_bytes(encoder.value)

            states = []
            start = start + period
        
def render(run_name=-1):
    path = paths.path(run_name, 'recording')
    if not path.exists():
        raise ValueError(f'No recording for "{run_name}"')
    pickle.loads(path.read_bytes())

