import torch
from rebar import arrdict
from megastep import modules, core, plotting
import matplotlib.pyplot as plt
from ... import toys, scene

class Minimal:
    """A minimal environment with no rewards or resets, just to demonstrate physics and rendering"""

    def __init__(self, *args, **kwargs):
        geometries = [toys.box()]
        scenery = scene.scenery(geometries)
        self.core = core.Core(scenery, *args, **kwargs)
        self.mover = modules.SimpleMovement(self.core)
        self.observer = modules.RGBD(self.core)
        self.respawner = modules.RandomSpawns(geometries, self.core)

        self.action_space = self.mover.space
        self.observation_space = self.observer.space

    @torch.no_grad()
    def reset(self):
        self.respawner(self.core.env_full(True).unsqueeze(-1))
        return arrdict.arrdict(
            obs=self.observer(),
            reward=self.core.env_full(0.),
            reset=self.core.env_full(True))

    @torch.no_grad()
    def step(self, decision):
        self.mover(decision)
        return arrdict.arrdict(
            obs=self.observer(),            
            reward=self.core.env_full(0.),
            reset=self.core.env_full(True))

    def state(self, d=0):
        return arrdict.arrdict(
            **self.core.state(d),
            obs=self.observer.state(d))

    @classmethod
    def plot_state(cls, state):
        fig = plt.figure()
        gs = plt.GridSpec(2, 2, fig, 0, 0, 1, 1)

        plotting.plot_core(state, plt.subplot(gs[:, 0]))
        plotting.plot_images(state.obs, [plt.subplot(gs[0, 1])])

        return fig

    def display(self, d=0):
        return self.plot_state(arrdict.numpyify(self.state(d)))


