from .simulator import Simulator
from . import modules
import torch
from rebar import arrdict
from functools import wraps

class MinimalEnv(Simulator, modules.SimpleMovement, modules.RGBObserver):
    """A minimal environment with no rewards or resets, just to demonstrate physics and rendering"""

    @wraps(Simulator.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def reset(self):
        reset = self._full(True)
        self._respawn(reset)
        return arrdict(obs=self._observe(), reset=reset, terminal=self._full(False), reward=self._full(0.).float())

    @torch.no_grad()
    def step(self, decisions):
        self._move(decisions)
        self._physics()

        reset = self._full(False)
        self._respawn(reset)
        return arrdict(obs=self._observe(), reset=reset, terminal=self._full(False), reward=self._full(False))