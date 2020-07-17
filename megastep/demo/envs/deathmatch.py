import torch
from megastep import modules, core, plotting, spaces, scene, cubicasa
from rebar import arrdict, dotdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

CLEARANCE = 1.

@dotdict.mapping
def expand(x):
    B, A = x.shape[:2]
    return x.reshape(B*A, 1, *x.shape[2:])

@dotdict.mapping
def collapse(x, n_agents):
    B = x.shape[0]
    return x.reshape(B//n_agents, n_agents, *x.shape[2:])

class Deathmatch:

    def __init__(self, n_envs, n_agents, *args, **kwargs):
        geometries = cubicasa.sample(n_envs)
        scenery = scene.scenery(geometries, n_agents)
        self.core = core.Core(scenery, *args, res=4*128, fov=70, **kwargs)
        self.rgb = modules.RGB(self.core, n_agents=1, subsample=4)
        self.depth = modules.Depth(self.core, n_agents=1, subsample=4)
        self.imu = modules.IMU(self.core, n_agents=1)
        self.movement = modules.MomentumMovement(self.core, n_agents=1)
        self.spawner = modules.RandomSpawns(geometries, self.core)

        self.action_space = self.movement.space
        self.obs_space = dotdict.dotdict(
            rgb=self.rgb.space,
            d=self.depth.space,
            imu=self.imu.space,
            health=spaces.MultiVector(1, 1))

        self.bounds = arrdict.torchify(np.stack([g.masks.shape*g.res for g in geometries])).to(self.core.device)
        self.health = self.core.agent_full(np.nan)
        self.damage = self.core.agent_full(np.nan)

        self.n_envs = self.core.n_envs*self.core.n_agents
        self.device = self.core.device

    def _reset(self, reset=None):
        reset = (self.health <= 0) if reset is None else reset
        self.spawner(reset)
        self.health[reset] = 1.
        self.damage[reset] = 0.
        return reset.reshape(-1)

    def _shoot(self, opponents):
        res = opponents.size(-1)
        middle = slice(res//2-1, res//2+1)
        agents = torch.arange(self.core.n_agents, device=self.core.device)
        matchings = (opponents[:, :, None] == agents[None, None, :, None, None])[..., middle].any(-1).any(-1)
        self.matchings = matchings
        
        hits = matchings.sum(2).float()
        wounds = matchings.sum(1).float()

        self.damage[:] += .05*hits

        pos = self.core.agents.positions 
        outside = (pos < -CLEARANCE).any(-1) | (pos > (self.bounds[:, None] + CLEARANCE)).any(-1)

        # 5% damage per hit, .1% damage per timestep
        self.health[:] += -.05*(wounds + outside) - .001
        
        return hits.reshape(-1)

    def _observe(self):
        render = modules.render(self.core)
        line_idxs = modules.downsample(render.indices, self.rgb.subsample)[..., self.rgb.subsample//2]
        obj_idxs = line_idxs//len(self.core.scenery.model)
        mask = (0 <= line_idxs) & (obj_idxs < self.core.n_agents)
        opponents = obj_idxs.where(mask, torch.full_like(line_idxs, -1))
        hits = self._shoot(opponents)
        obs = arrdict.arrdict(
                rgb=self.rgb(render), 
                d=self.depth(render), 
                imu=self.imu(),
                health=self.health.unsqueeze(-1).clone())
        return obs, hits

    @torch.no_grad()
    def reset(self):
        reset = self._reset(self.core.agent_full(True))
        obs, reward = self._observe()
        return arrdict.arrdict(
            obs=expand(obs),
            reward=reward,
            reset=reset)

    @torch.no_grad()
    def step(self, decision):
        reset = self._reset()
        self.movement(collapse(decision, self.core.n_agents))
        obs, reward = self._observe()
        return arrdict.arrdict(
            obs=expand(obs),
            reward=reward,
            reset=reset)

    def state(self, e=0):
        return arrdict.arrdict(
            core=self.core.state(e),
            rgb=self.rgb.state(e),
            d=self.depth.state(e),
            health=self.health[e].clone(),
            damage=self.damage[e].clone(),
            matchings=self.matchings[e].clone(),
            bounds=self.bounds[e].clone())

    @classmethod
    def plot_state(cls, state):
        n_agents = state.core.n_agents
        show_value = 'decision' in state

        fig = plt.figure()
        gs = plt.GridSpec(n_agents, 4 if show_value else 3, fig)

        colors = [f'C{i}' for i in range(n_agents)]

        plan = core.Core.plot_state(state.core, plt.subplot(gs[:-1, :-1]))

        # Add hits
        origin, dest = state.matchings.nonzero()
        lines = state.core.agents.positions[np.stack([origin, dest], 1)]
        linecolors = np.array(colors)[origin]
        lines = mpl.collections.LineCollection(lines, color=linecolors, linewidth=1, alpha=.5)
        plan.add_collection(lines)

        # Add bounding box
        size = state.bounds[::-1] + 2*CLEARANCE
        bounds = mpl.patches.Rectangle(
            (-CLEARANCE, -CLEARANCE), *size, 
            linewidth=1, edgecolor='k', facecolor=(0., 0., 0., 0.))
        plan.add_artist(bounds)

        # Add observations
        images = {'rgb': state.rgb, 'd': state.d}
        plotting.plot_images(images, [plt.subplot(gs[i, -1]) for i in range(n_agents)])

        ax = plt.subplot(gs[-1, 0])
        ax.barh(np.arange(n_agents), state.health, color=colors)
        ax.set_ylabel('health')
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)

        ax = plt.subplot(gs[-1, 1])
        ax.barh(np.arange(n_agents), state.damage, color=colors)
        ax.set_ylabel('inflicted')
        ax.set_yticks([])
        ax.invert_yaxis()

        if show_value:
            ax = plt.subplot(gs[-1, 2])
            ax.barh(np.arange(n_agents), state.decision.value, color=colors)
            ax.set_ylabel('value')
            ax.set_yticks([])
            ax.invert_yaxis()

        return fig

    def display(self, e=0):
        return self.plot_state(arrdict.numpyify(self.state(e=e)))