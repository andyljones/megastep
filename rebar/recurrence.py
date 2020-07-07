from . import arrdict
from torch import nn
from contextlib import contextmanager

class State:

    def __init__(self):
        super().__init__()

        self._value = None

    def get(self, factory=None):
        if self._value is None and factory is not None:
            self._value = factory()
        return self._value

    def set(self, value):
        self._value = value

    def clear(self):
        self._value = None

    def __repr__(self):
        return f'State({self._value})'
    
    def __str__(self):
        return repr(self)

def states(net):
    substates = {k: states(v) for k, v in net.named_children()}
    ownstates = {k: getattr(net, k) for k in dir(net) if isinstance(getattr(net, k), State)}
    return arrdict.arrdict({k: v for k, v in {**ownstates, **substates}.items() if v})

def _nonnull(x):
    y = type(x)()
    for k, v in x.items():
        if isinstance(v, dict):
            subtree = _nonnull(v)
            if subtree:
                y[k] = subtree
        elif v is not None:
            y[k] = v
    return y

def get(net):
    return _nonnull(states(net).map(lambda s: s.get()))

def set(net, state):
    state.starmap(lambda r, n: n.set(r), states(net))

def clear(net):
    states(net).map(lambda s: s.clear())

@contextmanager
def temp_clear(net):
    original = get(net)
    clear(net)
    try:
        yield
    finally:
        set(net, original)

@contextmanager
def temp_set(net, state):
    original = get(net)
    set(net, state)
    try:
        yield
    finally:
        set(net, original)

@contextmanager
def temp_clear_set(net, state):
    with temp_clear(net), temp_set(net, state):
        yield net

class Sequential(nn.Sequential):

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input