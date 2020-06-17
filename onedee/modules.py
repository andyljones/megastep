import numpy as np
import torch
from rebar import arrdict
from rebar.arrdict import cat, stack, tensorify
from . import spaces, core, plotting
import matplotlib.pyplot as plt


def to_global_frame(agents, p):
    a = np.pi/180*agents.angles
    c, s = torch.cos(a), torch.sin(a)
    x, y = p[..., 0], p[..., 1]
    return torch.stack([c*x - s*y, s*x + c*y], -1)

class SimpleMovement:

    def __init__(self, core, *args, accel=10, ang_accel=180, **kwargs):
        # noop, forward/backward, strafe left/right, turn left/right
        self._core = core
        momenta = torch.tensor([[0., 0.], [0., 1.], [0.,-1.], [1., 0.], [-1.,0.], [0., 0.], [0., 0.]])
        angmomenta = torch.tensor([0., 0., 0., 0., 0., +1., -1.])
        self._actionset = arrdict(
            momenta=accel/core.fps*momenta,
            angmomenta=ang_accel/core.fps*angmomenta
        ).to(core.device)

        self.action_space = spaces.MultiDiscrete(core.n_agents, 7)

    def __call__(self, decisions):
        core = self._core
        delta = self._actionset[decisions.actions]
        core.agents.angmomenta[:] = delta.angmomenta
        core.agents.momenta[:] = to_global_frame(core.agents, delta.momenta)
        core.cuda.physics(core.scene, core.agents)

class MomentumMovement:

    def __init__(self, core, *args, accel=5, ang_accel=90, decay=.125, **kwargs):
        # noop, forward/backward, strafe left/right, turn left/right
        self._core = core
        momenta = torch.tensor([[0., 0.], [0., 1.], [0.,-1.], [1., 0.], [-1.,0.], [0., 0.], [0., 0.]])
        angmomenta = torch.tensor([0., 0., 0., 0., 0., +1., -1.])
        self._actionset = arrdict(
            momenta=accel/core.fps*momenta,
            angmomenta=ang_accel/core.fps*angmomenta
        ).to(core.device)

        self._decay = decay

        self.action_space = spaces.MultiDiscrete(core.n_agents, 7)

    def __call__(self, decisions):
        core = self._core
        delta = self._actionset[decisions.actions]
        core.agents.angmomenta[:] = (1 - self._decay)*core.agents.angmomenta + delta.angmomenta
        core.agents.momenta[:] = (1 - self._decay)*core.agents.momenta + to_global_frame(core.agents, delta.momenta)
        core.cuda.physics(core.scene, core.agents)


def unpack(d):
    if isinstance(d, torch.Tensor):
        return d
    return arrdict({k: unpack(getattr(d, k)) for k in dir(d) if not k.startswith('_')})
        
class RGBDObserver:

    def __init__(self, core, *args, max_depth=10, **kwargs):
        self._core = core
        self.observation_space = arrdict(
            rgb=spaces.MultiImage(core.n_agents, 3, 1, core.res),
            d=spaces.MultiImage(core.n_agents, 1, 1, core.res),)
        self._max_depth = max_depth

    def render(self):
        core = self._core
        render = unpack(core.cuda.render(core.agents, core.scene))
        render = arrdict({k: v.unsqueeze(2) for k, v in render.items()})
        render['screen'] = render.screen.permute(0, 1, 4, 2, 3)
        return render

    def _downsample(self, screen):
        core = self._core
        return screen.view(*screen.shape[:-1], screen.shape[-1]//core.supersample, core.supersample).mean(-1)

    def __call__(self, render=None):
        render = self.render() if render is None else render
        self._last_obs = arrdict(
            rgb=self._downsample(render.screen),
            d=1 - self._downsample(render.distances.div(self._max_depth).clamp(0, 1)).unsqueeze(3))
        return self._last_obs
    
    def state(self, d):
        return self._last_obs[d].clone()
        
class RandomSpawns(core.Core):

    def __init__(self, core, *args, n_spawns=100, **kwargs):
        self._core = core

        assert core.n_agents == 1
        self._n_spawns = n_spawns

        spawns = []
        for g in core.geometries:
            sample = np.stack((g.masks > 0).nonzero(), -1)
            sample = sample[core.random.choice(np.arange(len(sample)), n_spawns)]
            
            i, j = sample.T + .5
            xy = g.res*np.stack([j, g.masks.shape[0] - i], -1)

            spawns.append(arrdict({
                'positions': xy[:, None],
                'angles': core.random.uniform(-180, +180, (n_spawns, core.n_agents))}))

        self._spawns = tensorify(stack(spawns)).to(core.device)

    def __call__(self, reset):
        core = self._core
        required = reset.nonzero().squeeze(-1)
        choices = torch.randint_like(required, 0, self._n_spawns)
        core.agents.angles[required] = self._spawns.angles[required, choices] 
        core.agents.positions[required] = self._spawns.positions[required, choices] 
        core.agents.momenta[required] = 0.
        core.agents.angmomenta[required] = 0.

