import torch
from .. import modules, core, plotting
from rebar import arrdict
import matplotlib.pyplot as plt
import numpy as np

class Tag:

    def __init__(self, *args, **kwargs):
        self._core = core.Core(*args, **kwargs)
        self._rgbd = modules.RGBD(self._core)
        self._mover = modules.MomentumMovement(self._core)
        self._respawner = modules.RandomSpawns(self._core)

        self.action_space = self._mover.space
        self.observation_space = self._rgbd.space

    @torch.no_grad()
    def reset(self):
        self._respawner(self._core.env_full(True))
        return arrdict(
            obs=self._rgbd(),
            reward=self._core.env_full(0.),
            reset=self._core.env_full(True),
            terminal=self._core.env_full(False),)

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)
        return arrdict(
            obs=self._rgbd(),            
            reward=self._core.env_full(0.),
            reset=self._core.env_full(True),
            terminal=self._core.env_full(False),)

    def state(self, d=0):
        return arrdict(
            **self._core.state(d),
            obs=self._rgbd.state(d),)

    @classmethod
    def plot_state(cls, state, zoom=False):
        n_agents = len(state.agents.angles)

        fig = plt.figure()
        gs = plt.GridSpec(n_agents, 2, fig, 0, 0, 1, 1)

        ax = plotting.plot_core(state, plt.subplot(gs[:, 0]), zoom=zoom)

        images = {k: v for k, v in state.obs.items() if k != 'imu'}
        plotting.plot_images(images, [plt.subplot(gs[i, 1]) for i in range(n_agents)])

        return fig

