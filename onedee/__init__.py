from .core import Core
from . import modules
import torch
from rebar import arrdict
from functools import wraps

class MinimalEnv(modules.SimpleMovement, modules.RGBObserver, modules.RandomSpawns):
    """A minimal environment with no rewards or resets, just to demonstrate physics and rendering"""

    @wraps(Core.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def reset(self):
        self._respawn(self._full(True))
        return arrdict(obs=self._observe())

    @torch.no_grad()
    def step(self, decisions):
        self._move(decisions)
        return arrdict(obs=self._observe())
    
class ExplorationEnv(modules.SimpleMovement, modules.RGBObserver, modules.RandomSpawns):

    @wraps(Core.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def reset(self):
        reset = self._full(True)
        self._respawn(reset)
        return arrdict(obs=self._observe(), reset=reset, terminal=self._full(False), reward=self._full(0.))

    @torch.no_grad()
    def step(self, decisions):
        self._move(decisions)

        reset = self._full(False)
        self._respawn(reset)
        return arrdict(obs=self._observe(), reset=reset, terminal=self._full(False), reward=self._full(0.))
