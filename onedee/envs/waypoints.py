import numpy as np
import torch
from .. import modules, core, spaces, plotting
from rebar import arrdict
import matplotlib.pyplot as plt

class WaypointEnv: 

    def __init__(self, *args, max_length=512, **kwargs):
        self._core = core.Core(*args, **kwargs)
        self._mover = modules.SimpleMovement(self._core)
        self._observer = modules.RGBDObserver(self._core)
        self._respawner = modules.RandomSpawns(self._core)

        self.action_space = self._mover.action_space
        self.observation_space = arrdict(
            **self._observer.observation_space,
            waypoint=spaces.MultiVector(self._core.n_agents, 2))

        self._waypoints = torch.full((self._core.n_envs, self._core.n_agents, 2), np.nan, dtype=torch.float, device=self._core.device)

    def _refresh_waypoints(self, refresh):
        origin = self._core.agents.positions[refresh]
        self._waypoints[refresh] = origin + torch.randn_like(origin)

    def _reset(self, reset):
        self._respawner(reset)
        self._refresh_waypoints(reset[:, None])
    
    def _observe(self):
        obs = self._observer().copy()
        delta = self._waypoints - self._core.agents.positions
        relative = modules.to_local_frame(self._core.agents.angles, delta)
        obs['waypoint'] = relative
        return obs.clone()

    def _success(self):
        distances = (self._waypoints - self._core.agents.positions).pow(2).sum(-1).pow(.5)
        success = (distances < .15).all(-1)
        failure = (distances > 5).any(-1)
        return success, failure

    @torch.no_grad()
    def reset(self):
        reset = core.env_full_like(self._core, True)
        self._reset(reset)
        return arrdict(
            obs=self._observe(),
            reward=core.env_full_like(self._core, 0.),
            reset=reset,
            terminal=reset)

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)
        success, failure = self._success()
        self._reset(success | failure)
        return arrdict(
            obs=self._observe(),
            reward=success.float(),
            reset=success | failure,
            terminal=success | failure)

    def state(self, d=0):
        return arrdict(
            **self._core.state(d),
            obs=self._observer.state(d),
            waypoint=self._waypoints[d].clone())

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
