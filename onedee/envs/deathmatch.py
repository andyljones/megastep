import torch
from .. import modules, core, plotting
from rebar.arrdict import mapping
from rebar import arrdict
import matplotlib.pyplot as plt
import numpy as np

@mapping
def expand(x):
    B, A = x.shape[:2]
    return x.reshape(B*A, 1, *x.shape[2:])

@mapping
def collapse(x):
    B = x.shape[0]
    return x.reshape(B//2, 2, *x.shape[2:])

class Deathmatch:

    def __init__(self, *args, **kwargs):
        self._core = core.Core(*args, **kwargs)
        self._rgbd = modules.RGBD(self._core, n_agents=1)
        self._mover = modules.MomentumMovement(self._core, n_agents=1)
        self._respawner = modules.RandomSpawns(self._core)
        self._lengths = modules.RandomLengths(self._core)

        self.action_space = self._mover.space
        self.observation_space = self._rgbd.space

    def _reset(self, reset=None):
        reset = self._lengths(reset)
        self._respawner(reset)
        return reset[:, None].repeat_interleave(2, 1).reshape(-1)

    def _downsample(self, screen):
        core = self._core
        idx = core.supersample//2
        return screen.view(*screen.shape[:-1], screen.shape[-1]//core.supersample, core.supersample)[..., idx]

    def _reward(self, opponents):
        agents = torch.arange(self._core.n_agents, device=self._core.device)
        matchings = (opponents[:, :, None] == agents[None, None, :, None, None]).any(-1).any(-1)
        
        success = matchings.sum(2)
        failures = matchings.sum(1)

        return (.5*success.float() - failures.float()).reshape(-1)

    def _observe(self):
        render = self._rgbd.render()
        indices = self._downsample(render.indices)
        obj = indices//len(self._core.scene.frame)
        mask = (0 <= indices) & (obj < self._core.n_agents)
        opponents = obj.where(mask, torch.full_like(indices, -1))
        return arrdict(**self._rgbd(render)), opponents

    @torch.no_grad()
    def reset(self):
        reset = self._reset(self._core.env_full(True))
        obs, opponents = self._observe()
        return arrdict(
            obs=expand(obs),
            reward=self._reward(opponents),
            reset=reset,
            terminal=reset,)

    @torch.no_grad()
    def step(self, decision):
        reset = self._reset()
        self._mover(collapse(decision))
        obs, opponents = self._observe()
        return arrdict(
            obs=expand(obs),
            reward=self._reward(opponents),
            reset=reset,
            terminal=reset,)

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

