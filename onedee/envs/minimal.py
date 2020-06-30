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

class Minimal:
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

