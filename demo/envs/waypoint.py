import numpy as np
import torch
from onedee import modules, core, spaces, plotting
from rebar import arrdict
import matplotlib.pyplot as plt

class RandomGoals:

    def __init__(self, core, *args, n_goals=10, **kwargs):
        self._core = core

        self._n_goals = n_goals
        self._goals = arrdict.tensorify(modules.random_empty_positions(core, n_goals)).to(core.device)
        self.current = torch.full_like(self._goals[:, 0], np.nan)

    def __call__(self, reset, distance, temperature=10, clip=2):
        if not reset.any():
            return self._goals.new_empty((0, *self._goals.shape[2:]))
        d = (self._goals[reset] - self._core.agents.positions[reset, None]).pow(2).sum(-1).pow(.5).mean(-1)

        logits = -(d/distance).log10().clamp(-clip, +clip).abs().mul(temperature)
        sample = torch.distributions.Categorical(logits=logits).sample()

        self.current[reset] = self._goals[reset.nonzero().squeeze(-1), sample]

class Waypoint: 

    def __init__(self, *args, max_length=512, **kwargs):
        self._core = core.Core(*args, **kwargs)
        self._mover = modules.MomentumMovement(self._core)
        self._rgbd = modules.RGBD(self._core)
        self._respawner = modules.RandomSpawns(self._core)
        self._lengths = modules.RandomLengths(self._core, max_length)
        self._goals = RandomGoals(self._core)

        self.action_space = self._mover.space
        self.observation_space = arrdict(
            **self._rgbd.space,
            waypoint=spaces.MultiVector(self._core.n_agents, 2))

    def _reset(self, reset):
        reset = self._lengths(reset)
        self._respawner(reset)
        self._goals(reset, 1.)
        return reset
    
    def _observe(self):
        obs = self._rgbd().copy()
        delta = self._goals.current - self._core.agents.positions
        relative = modules.to_local_frame(self._core.agents.angles, delta)
        obs['waypoint'] = relative
        return obs.clone()

    @torch.no_grad()
    def reset(self):
        reset = self._core.env_full(True)
        reset = self._reset(reset)
        return arrdict(
            obs=self._observe(),
            reward=self._core.env_full(0.),
            reset=reset,
            terminal=reset)

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)
        distances = (self._goals.current - self._core.agents.positions).pow(2).sum(-1).pow(.5)
        success = distances < .15
        reset = self._reset(success.all(-1))
        return arrdict(
            obs=self._observe(),
            reward=success.sum(-1).float(),
            reset=reset,
            terminal=reset)

    def state(self, d=0):
        return arrdict(
            **self._core.state(d),
            obs=self._rgbd.state(d),
            waypoint=self._goals.current[d].clone())

    @classmethod
    def plot_state(cls, state):
        fig = plt.figure()
        gs = plt.GridSpec(2, 2, fig, 0, 0, 1, 1)

        ax = plotting.plot_core(state, plt.subplot(gs[:, 0]))
        plotting.plot_images(state.obs, [plt.subplot(gs[0, 1])])

        ax.scatter(*state.waypoint.T, marker='x', color='red')

        return fig

    def display(self, d=0):
        return self.plot_state(arrdict.numpyify(self.state(d)))

    def decide(self, world):
        accel = self._mover._actionset.momenta
        actions = (world.obs.waypoint[..., None, :]*accel).sum(-1).argmax(-1)
        return arrdict(actions=actions)

class PointGoal:

    def __init__(self, *args, max_length=512, **kwargs):
        self._core = core.Core(*args, **kwargs)
        self._mover = modules.MomentumMovement(self._core)
        self._rgbd = modules.RGBD(self._core)
        self._imu = modules.IMU(self._core)
        self._respawner = modules.RandomSpawns(self._core)
        self._lengths = modules.RandomLengths(self._core, max_length)
        self._goals = RandomGoals(self._core)

        self._spawns = arrdict(
            angles=torch.zeros_like(self._core.agents.angles),
            positions=torch.zeros_like(self._core.agents.positions))

        self.action_space = self._mover.space
        self.observation_space = arrdict(
            **self._rgbd.space,
            imu=self._imu.space,
            waypoint=spaces.MultiVector(self._core.n_agents, 2))

    def _reset(self, reset):
        reset = self._lengths(reset)
        self._respawner(reset)

        self._spawns.angles[reset] = self._core.agents.angles[reset]
        self._spawns.positions[reset] = self._core.agents.positions[reset]

        self._goals(reset, 1.)
        return reset
    
    def _observe(self):
        obs = self._rgbd().copy()
        delta = self._goals.current - self._spawns.positions
        relative = modules.to_local_frame(self._spawns.angles, delta)
        obs['waypoint'] = relative
        obs['imu'] = self._imu()
        return obs.clone()

    @torch.no_grad()
    def reset(self):
        reset = self._core.env_full(True)
        reset = self._reset(reset)
        return arrdict(
            obs=self._observe(),
            reward=self._core.env_full(0.),
            reset=reset,
            terminal=self._core.env_full(False))

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)
        distances = (self._goals.current - self._core.agents.positions).pow(2).sum(-1).pow(.5)
        success = distances < .15
        reset = self._reset(success.all(-1))
        return arrdict(
            obs=self._observe(),
            reward=success.sum(-1).float(),
            reset=reset,
            terminal=reset)

    def state(self, d=0):
        return arrdict(
            **self._core.state(d),
            obs=self._rgbd.state(d),
            waypoint=self._goals.current[d].clone())

    @classmethod
    def plot_state(cls, state):
        fig = plt.figure()
        gs = plt.GridSpec(2, 2, fig, 0, 0, 1, 1)

        ax = plotting.plot_core(state, plt.subplot(gs[:, 0]))
        plotting.plot_images(state.obs, [plt.subplot(gs[0, 1])])

        ax.scatter(*state.waypoint.T, marker='x', color='red')

        return fig

    def display(self, d=0):
        return self.plot_state(arrdict.numpyify(self.state(d)))

    def decide(self, world):
        accel = self._mover._actionset.momenta
        actions = (world.obs.waypoint[..., None, :]*accel).sum(-1).argmax(-1)
        return arrdict(actions=actions)