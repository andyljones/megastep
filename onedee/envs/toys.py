import torch
from rebar import arrdict
from .. import spaces, modules, core

class IndicatorEnv:

    def __init__(self, n_envs=1, n_agents=1):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.observation_space = spaces.MultiVector(self.n_agents, 1)
        self.action_space = spaces.MultiDiscrete(self.n_agents, 2)
        self.device = torch.device('cuda')

    def _observe(self):
        self._last_obs = torch.rand((self.n_envs, self.n_agents, 1), device=self.device).gt(.5).float()
        return self._last_obs

    @torch.no_grad()
    def reset(self):
        return arrdict(
            obs=self._observe(),
            reward=torch.full((self.n_envs,), 0., dtype=torch.float, device=self.device),
            reset=torch.full((self.n_envs,), True, device=self.device, dtype=torch.bool),
            terminal=torch.full((self.n_envs,), True, device=self.device, dtype=torch.bool))

    @torch.no_grad()
    def step(self, decisions):
        reward = (decisions.actions == self._last_obs[..., 0]).float().sum(-1)
        return arrdict(
            obs=self._observe(),
            reward=reward,
            reset=torch.full((self.n_envs,), False, device=self.device, dtype=torch.bool),
            terminal=torch.full((self.n_envs,), False, device=self.device, dtype=torch.bool))

class MinimalEnv:
    """A minimal environment with no rewards or resets, just to demonstrate physics and rendering"""

    def __init__(self, *args, **kwargs):
        self._core = core.Core(*args, **kwargs)
        self._mover = modules.SimpleMovement(self._core)
        self._observer = modules.RGBDObserver(self._core)
        self._respawner = modules.RandomSpawns(self._core)

        self.action_space = self._mover.action_space
        self.observation_space = self._observer.observation_space

    @torch.no_grad()
    def reset(self):
        self._respawner(core.env_full_like(self._core, True))
        return arrdict(
            obs=self._observer(),
            reward=torch.full((self.n_envs,), 0., dtype=torch.float, device=self.core.device),
            reset=torch.full((self.n_envs,), True, dtype=torch.bool, device=self.core.device),
            terminal=torch.full((self.n_envs,), True, dtype=torch.bool, device=self.core.device))

    @torch.no_grad()
    def step(self, decisions):
        self._mover(decisions)
        return arrdict(
            obs=self._observer(),            
            reward=torch.full((self.n_envs,), 0., dtype=torch.float, device=self.core.device),
            reset=torch.full((self.n_envs,), False, dtype=torch.bool, device=self.core.device),
            terminal=torch.full((self.n_envs,), False, dtype=torch.bool, device=self.core.device))
