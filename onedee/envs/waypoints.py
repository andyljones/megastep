import numpy as np
import torch
from .. import modules, core, spaces
from rebar import arrdict

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

        self._waypoints = torch.full((self._core.n_envs, self._core.n_agents, 2), np.nan, dtype=torch.float, device=self.core.device)

    def _refresh_waypoints(self, refresh):
        origin = self._core.agents.positions[refresh]
        self._waypoints[refresh] = origin + torch.randn_like(origin)

    def _reset(self, reset):
        self._respawner(reset)
        self._refresh_waypoints(reset[:, None])
    
    def _observe(self):
        obs = self._observer(),
        delta = self._waypoints - self._core.agents.positions
        relative = modules.to_local_frame(self._core.agents.angles, delta)
        obs['waypoint'] = relative
        return obs

    def _reward(self):
        distances = (self._waypoints - self._core.agents.positions).pow(2).sum(-1)
        success = (distances < .15)
        self._refresh_waypoints(success)
        return success.float()

    @torch.no_grad()
    def reset(self):
        self._reset(core.env_full_like(self._core, True))
        return arrdict(
            obs=self._observe(),
            reward=self._reward(),
            reset=torch.full((self.n_envs,), False, dtype=torch.bool, device=self.core.device),
            terminal=torch.full((self.n_envs,), False, dtype=torch.bool, device=self.core.device))

    @torch.no_grad()
    def step(self, decisions):
        self._mover(decisions)
        return arrdict(
            obs=self._observe(),
            reward=self._reward(),
            reset=torch.full((self.n_envs,), False, dtype=torch.bool, device=self.core.device),
            terminal=torch.full((self.n_envs,), False, dtype=torch.bool, device=self.core.device))

