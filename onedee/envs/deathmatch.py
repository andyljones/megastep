import torch
from .. import modules, core, plotting, spaces
from rebar.arrdict import mapping
from rebar import arrdict, dotdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

CLEARANCE = 1.

@mapping
def expand(x):
    B, A = x.shape[:2]
    return x.reshape(B*A, 1, *x.shape[2:])

@mapping
def collapse(x, n_agents):
    B = x.shape[0]
    return x.reshape(B//n_agents, n_agents, *x.shape[2:])

class Deathmatch:

    def __init__(self, *args, **kwargs):
        self._core = core.Core(*args, res=128, fov=60, supersample=4, **kwargs)
        self._rgbd = modules.RGBD(self._core, n_agents=1)
        self._imu = modules.IMU(self._core, n_agents=1)
        self._mover = modules.MomentumMovement(self._core, n_agents=1)
        self._respawner = modules.RandomSpawns(self._core)

        self.action_space = self._mover.space
        self.observation_space = dotdict(
            **self._rgbd.space,
            imu=self._imu.space,
            health=spaces.MultiVector(1, 1))

        self._bounds = arrdict.tensorify(np.stack([g.masks.shape*g.res for g in self._core.geometries])).to(self._core.device)
        self._health = self._core.agent_full(np.nan)
        self._damage = self._core.agent_full(np.nan)

        self.n_envs = self._core.n_envs*self._core.n_agents
        self.device = self._core.device

    def _reset(self, reset=None):
        reset = (self._health <= 0) if reset is None else reset
        self._respawner(reset)
        self._health[reset] = 1.
        self._damage[reset] = 0.
        return reset.reshape(-1)

    def _downsample(self, screen):
        core = self._core
        idx = core.supersample//2
        return screen.view(*screen.shape[:-1], screen.shape[-1]//core.supersample, core.supersample)[..., idx]

    def _shoot(self, opponents):
        res = opponents.size(-1)
        middle = slice(res//2-1, res//2+1)
        agents = torch.arange(self._core.n_agents, device=self._core.device)
        matchings = (opponents[:, :, None] == agents[None, None, :, None, None])[..., middle].any(-1).any(-1)
        self._matchings = matchings
        
        hits = matchings.sum(2).float()
        wounds = matchings.sum(1).float()

        self._damage[:] += .05*hits

        pos = self._core.agents.positions 
        outside = (pos < -CLEARANCE).any(-1) | (pos > (self._bounds[:, None] + CLEARANCE)).any(-1)

        self._health[:] += -.05*(wounds + outside) - .001
        
        return hits.reshape(-1)

    def _observe(self):
        render = self._rgbd.render()
        indices = self._downsample(render.indices)
        obj = indices//len(self._core.scene.frame)
        mask = (0 <= indices) & (obj < self._core.n_agents)
        opponents = obj.where(mask, torch.full_like(indices, -1))
        hits = self._shoot(opponents)
        return arrdict(
            **self._rgbd(render), 
            imu=self._imu(),
            health=self._health.unsqueeze(-1).clone()), hits

    @torch.no_grad()
    def reset(self):
        reset = self._reset(self._core.agent_full(True))
        obs, reward = self._observe()
        return arrdict(
            obs=expand(obs),
            reward=reward,
            reset=reset,
            terminal=reset,)

    @torch.no_grad()
    def step(self, decision):
        reset = self._reset()
        self._mover(collapse(decision, self._core.n_agents))
        obs, reward = self._observe()
        return arrdict(
            obs=expand(obs),
            reward=reward,
            reset=reset,
            terminal=reset,)

    def state(self, e=0):
        return arrdict(
            **self._core.state(e),
            obs=self._rgbd.state(e),
            health=self._health[e].clone(),
            damage=self._damage[e].clone(),
            matchings=self._matchings[e].clone(),
            bounds=self._bounds[e].clone())

    @classmethod
    def plot_state(cls, state, zoom=False):
        n_agents = len(state.agents.angles)
        show_value = 'decision' in state

        fig = plt.figure()
        gs = plt.GridSpec(n_agents, 4 if show_value else 3, fig)

        colors = [f'C{i}' for i in range(state.n_agents)]

        plan = plotting.plot_core(state, plt.subplot(gs[:-1, :-1]), zoom=zoom)

        # Add hits
        origin, dest = state.matchings.nonzero()
        lines = state.agents.positions[np.stack([origin, dest], 1)]
        linecolors = np.array(colors)[origin]
        lines = mpl.collections.LineCollection(lines, color=linecolors, linewidth=1)
        plan.add_collection(lines)

        # Add bounding box
        size = state.bounds[::-1] + 2*CLEARANCE
        bounds = mpl.patches.Rectangle(
            (-CLEARANCE, -CLEARANCE), *size, 
            linewidth=1, edgecolor='k', facecolor=(0., 0., 0., 0.))
        plan.add_artist(bounds)

        # Add observations
        images = {k: v for k, v in state.obs.items() if k != 'imu'}
        plotting.plot_images(images, [plt.subplot(gs[i, -1]) for i in range(n_agents)])

        ax = plt.subplot(gs[-1, 0])
        ax.barh(np.arange(state.n_agents), state.health, color=colors)
        ax.set_ylabel('health')
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)

        ax = plt.subplot(gs[-1, 1])
        ax.barh(np.arange(state.n_agents), state.damage, color=colors)
        ax.set_ylabel('inflicted')
        ax.set_yticks([])
        ax.invert_yaxis()

        if show_value:
            ax = plt.subplot(gs[-1, 2])
            ax.barh(np.arange(state.n_agents), state.decision.value, color=colors)
            ax.set_ylabel('value')
            ax.set_yticks([])
            ax.invert_yaxis()

        return fig

    def display(self, e=0):
        return self.plot_state(arrdict.numpyify(self.state(e=e)))