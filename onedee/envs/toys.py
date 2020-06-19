import torch
from rebar import arrdict
from .. import spaces, modules, core, plotting
import matplotlib.pyplot as plt

def falses(n_envs, device):
    return torch.zeros((n_envs,), dtype=torch.bool, device=device)

def trues(n_envs, device):
    return torch.ones((n_envs,), dtype=torch.bool, device=device)

def zeros(n_envs, device):
    return torch.ones((n_envs,), dtype=torch.float, device=device)

class ObliviousEnv:

    def __init__(self, n_envs=1, n_agents=1):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.observation_space = spaces.MultiEmpty()
        self.action_space = spaces.MultiConstant(self.n_agents)
        self.device = torch.device('cuda')

    @torch.no_grad()
    def reset(self):
        return arrdict(
            obs=torch.empty((self.n_envs, self.n_agents, 0), device=self.device),
            reward=zeros(self.n_envs, self.device),
            reset=trues(self.n_envs, self.device),
            terminal=trues(self.n_envs, self.device),)

    @torch.no_grad()
    def step(self, decision):
        return arrdict(
            obs=torch.empty((self.n_envs, self.n_agents, 0), device=self.device),
            reward=zeros(self.n_envs, self.device),
            reset=falses(self.n_envs, self.device),
            terminal=falses(self.n_envs, self.device),)

class BlindEnv:

    def __init__(self, n_envs=1, n_agents=1):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.observation_space = spaces.MultiEmpty()
        self.action_space = spaces.MultiDiscrete(self.n_agents, 2)
        self.device = torch.device('cuda')

    @torch.no_grad()
    def reset(self):
        return arrdict(
            obs=torch.zeros((self.n_envs, self.n_agents, 0), device=self.device),
            reward=zeros(self.n_envs, self.device),
            reset=trues(self.n_envs, self.device),
            terminal=trues(self.n_envs, self.device),)

    @torch.no_grad()
    def step(self, decision):
        reward = (decision.actions == 1).float().sum(-1)
        return arrdict(
            obs=torch.zeros((self.n_envs, self.n_agents, 0), device=self.device),
            reward=reward,
            reset=falses(self.n_envs, self.device),
            terminal=falses(self.n_envs, self.device),)

    def decide(self, world):
        return arrdict(actions=torch.ones_like(world.obs[..., 0]).int())


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
            reward=zeros(self.n_envs, self.device),
            reset=trues(self.n_envs, self.device),
            terminal=trues(self.n_envs, self.device),)

    @torch.no_grad()
    def step(self, decision):
        reward = (decision.actions == self._last_obs[..., 0]).float().sum(-1)
        return arrdict(
            obs=self._observe(),
            reward=reward,
            reset=falses(self.n_envs, self.device),
            terminal=falses(self.n_envs, self.device),)

    def decide(self, world):
        return arrdict(actions=world.obs[..., 0].int())


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
            reward=zeros(self.n_envs, self.device),
            reset=trues(self.n_envs, self.device),
            terminal=trues(self.n_envs, self.device),)

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)
        return arrdict(
            obs=self._observer(),            
            reward=zeros(self.n_envs, self.device),
            reset=falses(self.n_envs, self.device),
            terminal=falses(self.n_envs, self.device),)

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


