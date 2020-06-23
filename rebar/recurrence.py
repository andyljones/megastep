from .arrdict import arrdict
from torch import nn

class State:

    def __init__(self):
        super().__init__()

        self._ready = False
        self._value = None

    def get(self, factory=None):
        if not self._ready and factory is not None:
            self._value = factory()
        return self._value

    def set(self, value):
        self._ready = True
        self._value = value

    def clear(self):
        self._ready = False
        self._value = None

    def __repr__(self):
        return f'State({self._ready}, {self._value})'
    
    def __str__(self):
        return repr(self)

def state(net):
    substates = {k: state(v) for k, v in net.named_children()}
    states = {k: getattr(net, k) for k in dir(net) if isinstance(getattr(net, k), State)}
    combined = {**states, **substates}
    return arrdict({k: v for k, v in combined.items() if v})

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
    return _nonnull(state(net).map(lambda s: s.get()))

def set(net, recurrent):
    state(net).set(recurrent).starmap(lambda n, r: n.set(r), recurrent)

def clear(net):
    state(net).map(lambda s: s.clear())

class Sequential(nn.Sequential):

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input

