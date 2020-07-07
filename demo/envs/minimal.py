import torch
from rebar import arrdict
from megastep import modules, core, plotting
import matplotlib.pyplot as plt

class Minimal:
    """A minimal environment with no rewards or resets, just to demonstrate physics and rendering"""

    def __init__(self, *args, **kwargs):
        self._core = core.Core(*args, **kwargs)
        self._mover = modules.SimpleMovement(self._core)
        self._observer = modules.RGBD(self._core)
        self._respawner = modules.RandomSpawns(self._core)

        self.action_space = self._mover.space
        self.observation_space = self._observer.space

    @torch.no_grad()
    def reset(self):
        self._respawner(self._core.env_full(True))
        return arrdict(
            obs=self._observer(),
            reward=self._core.env_full(0.),
            reset=self._core.env_full(True),
            terminal=self._core.env_full(False),)

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)
        return arrdict(
            obs=self._observer(),            
            reward=self._core.env_full(0.),
            reset=self._core.env_full(True),
            terminal=self._core.env_full(False),)

    def state(self, d=0):
        return arrdict(
            **self._core.state(d),
            obs=self._observer.state(d))

    @classmethod
    def plot_state(cls, state):
        fig = plt.figure()
        gs = plt.GridSpec(2, 2, fig, 0, 0, 1, 1)

        plotting.plot_core(state, plt.subplot(gs[:, 0]))
        plotting.plot_images(state.obs, [plt.subplot(gs[0, 1])])

        return fig

    def display(self, d=0):
        return self.plot_state(arrdict.numpyify(self.state(d)))


