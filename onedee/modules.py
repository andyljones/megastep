import numpy as np
import torch
from rebar import arrdict
from rebar.arrdict import cat, stack, tensorify
from . import spaces, plotting
import matplotlib.pyplot as plt

def to_local_frame(angles, p):
    a = np.pi/180*angles
    c, s = torch.cos(a), torch.sin(a)
    x, y = p[..., 0], p[..., 1]
    return torch.stack([c*x + s*y, -s*x + c*y], -1)

def to_global_frame(angles, p):
    a = np.pi/180*angles
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

        self.space = spaces.MultiDiscrete(core.n_agents, 7)

    def __call__(self, decision):
        core = self._core
        delta = self._actionset[decision.actions]
        core.agents.angmomenta[:] = delta.angmomenta
        core.agents.momenta[:] = to_global_frame(core.agents.angles, delta.momenta)
        core.cuda.physics(core.scene, core.agents, core.progress)

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

        self.space = spaces.MultiDiscrete(core.n_agents, 7)

    def __call__(self, decision):
        core = self._core
        delta = self._actionset[decision.actions]
        core.agents.angmomenta[:] = (1 - self._decay)*core.agents.angmomenta + delta.angmomenta
        core.agents.momenta[:] = (1 - self._decay)*core.agents.momenta + to_global_frame(core.agents.angles, delta.momenta)
        core.cuda.physics(core.scene, core.agents, core.progress)

def unpack(d):
    if isinstance(d, torch.Tensor):
        return d
    return arrdict({k: unpack(getattr(d, k)) for k in dir(d) if not k.startswith('_')})
        
class RGBD:

    def __init__(self, core, *args, max_depth=10, **kwargs):
        self._core = core
        self.space = arrdict(
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
        depth = ((render.distances - self._core.agent_radius)/self._max_depth).clamp(0, 1)
        self._last_obs = arrdict(
            rgb=self._downsample(render.screen),
            d=self._downsample(depth).unsqueeze(3))
        return self._last_obs
    
    def state(self, d):
        return self._last_obs[d].clone()

class IMU:

    def __init__(self, core):
        self._core = core
        self.space = spaces.MultiVector(core.n_agents, 3)

    def __call__(self):
        return torch.cat([
            self._core.agents.angmomenta[..., None]/360.,
            to_local_frame(self._core.agents.angles, self._core.agents.momenta)/10.], -1)

def to_center_coords(indices, shape, res):
    i, j = indices[..., 0] + .5, indices[..., 1]
    xy = res*np.stack([j, shape[0] - i], -1)
    return xy

def random_empty_positions(core, n_points):
    points = []
    for g in core.geometries:
        sample = np.stack((g.masks > 0).nonzero(), -1)

        # There might be fewer open points than we're asking for
        n_possible = min(len(sample)//core.n_agents, n_points)
        sample = sample[core.random.choice(np.arange(len(sample)), (n_possible, core.n_agents), replace=True)]

        # So repeat the sample until we've got enough
        sample = np.concatenate([sample]*int(n_points/len(sample)+1))[-n_points:]
        sample = np.random.permutation(sample)
        points.append(to_center_coords(sample, g.masks.shape, g.res))
    return stack(points)
        
class RandomSpawns:

    def __init__(self, core, *args, n_spawns=100, **kwargs):
        self._core = core

        positions = random_empty_positions(core, n_spawns)
        angles = core.random.uniform(-180, +180, (len(core.geometries), n_spawns, core.n_agents))
        self._spawns = tensorify(arrdict(positions=positions, angles=angles)).to(core.device)

    def __call__(self, reset):
        core = self._core
        required = reset.nonzero().squeeze(-1)
        choices = torch.randint_like(required, 0, self._spawns.angles.shape[1])
        core.agents.angles[required] = self._spawns.angles[required, choices] 
        core.agents.positions[required] = self._spawns.positions[required, choices] 
        core.agents.momenta[required] = 0.
        core.agents.angmomenta[required] = 0.

class RandomLengths:

    def __init__(self, core, max_length=512, min_length=None):
        min_length = max_length//2 if min_length is None else min_length
        self._max_lengths = torch.randint(min_length, max_length, (core.n_envs,), dtype=torch.int, device=core.device)
        self._lengths = torch.zeros_like(self._max_lengths)
    
    def __call__(self, reset=None):
        self._lengths += 1
        reset = torch.zeros_like(self._lengths, dtype=torch.bool) if reset is None else reset
        reset = (self._lengths >= self._max_lengths) | reset
        self._lengths[reset] = 0
        return reset

    def state(self, d):
        return arrdict(length=self._lengths[d], max_length=self._max_lengths[d]).clone()
