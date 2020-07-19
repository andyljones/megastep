"""TODO-DOCS Module docs""" 

import numpy as np
import torch
from rebar import arrdict
from . import spaces, geometry, cuda, plotting
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

    def __init__(self, core, *args, accel=10, ang_accel=180, n_agents=None, **kwargs):
        # noop, forward/backward, strafe left/right, turn left/right
        self.core = core
        velocity = torch.tensor([[0., 0.], [0., 1.], [0.,-1.], [1., 0.], [-1.,0.], [0., 0.], [0., 0.]])
        angvelocity = torch.tensor([0., 0., 0., 0., 0., +1., -1.])
        self._actionset = arrdict.arrdict(
            velocity=accel/core.fps*velocity,
            angvelocity=ang_accel/core.fps*angvelocity
        ).to(core.device)

        self.space = spaces.MultiDiscrete(n_agents or core.n_agents, 7)

    def __call__(self, decision):
        core = self.core
        delta = self._actionset[decision.actions.long()]
        core.agents.angvelocity[:] = delta.angvelocity
        core.agents.velocity[:] = to_global_frame(core.agents.angles, delta.velocity)
        return cuda.physics(core.scenery, core.agents)

class MomentumMovement:

    def __init__(self, core, *args, accel=5, ang_accel=180, decay=.125, n_agents=None, **kwargs):
        # noop, forward/backward, strafe left/right, turn left/right
        self.core = core
        velocity = torch.tensor([[0., 0.], [0., 1.], [0.,-1.], [1., 0.], [-1.,0.], [0., 0.], [0., 0.]])
        angvelocity = torch.tensor([0., 0., 0., 0., 0., +1., -1.])
        self._actionset = arrdict.arrdict(
            velocity=accel/core.fps*velocity,
            angvelocity=ang_accel/core.fps*angvelocity
        ).to(core.device)

        self._decay = decay

        self.space = spaces.MultiDiscrete(n_agents or core.n_agents, 7)

    def __call__(self, decision):
        core = self.core
        delta = self._actionset[decision.actions.long()]
        core.agents.angvelocity[:] = (1 - self._decay)*core.agents.angvelocity + delta.angvelocity
        core.agents.velocity[:] = (1 - self._decay)*core.agents.velocity + to_global_frame(core.agents.angles, delta.velocity)
        return cuda.physics(core.scenery, core.agents)

def unpack(d):
    if isinstance(d, torch.Tensor):
        return d
    return arrdict.arrdict({k: unpack(getattr(d, k)) for k in dir(d) if not k.startswith('_')})

def render(core):
    render = unpack(cuda.render(core.scenery, core.agents))
    render = arrdict.arrdict({k: v.unsqueeze(2) for k, v in render.items()})
    render['screen'] = render.screen.permute(0, 1, 4, 2, 3)
    return render

def downsample(screen, subsample):
    return screen.view(*screen.shape[:-1], screen.shape[-1]//subsample, subsample)

class Depth:

    def __init__(self, core, *args, n_agents=None, subsample=1, max_depth=10, **kwargs):
        n_agents = n_agents or core.n_agents
        self.core = core
        self.space = spaces.MultiImage(n_agents, 1, 1, core.res//subsample)
        self.max_depth = max_depth
        self.subsample = subsample

    def __call__(self, r=None):
        r = render(self.core) if r is None else r
        depth = ((r.distances - self.core.agent_radius)/self.max_depth).clamp(0, 1)
        self._last_obs = downsample(depth, self.subsample).mean(-1).unsqueeze(3)
        return self._last_obs
    
    def state(self, e):
        return self._last_obs[e].clone()

class RGB:

    def __init__(self, core, *args, n_agents=None, subsample=1, **kwargs):
        n_agents = n_agents or core.n_agents
        self.core = core
        self.space = spaces.MultiImage(n_agents, 3, 1, core.res//subsample)
        self.subsample = subsample

    def __call__(self, r=None):
        r = render(self.core) if r is None else r
        self._last_obs = downsample(r.screen, self.subsample).mean(-1)
        return self._last_obs
    
    def state(self, e):
        return self._last_obs[e].clone()

    @classmethod
    def plot_state(cls, state, axes=None):
        n_agents = state.shape[0]
        axes = plt.subplots(n_agents, 1, squeeze=False) if axes is None else axes
        plotting.plot_images({'rgb': state}, axes)
        return axes

class IMU:

    def __init__(self, core, n_agents=None):
        self.core = core
        self.space = spaces.MultiVector(n_agents or core.n_agents, 3)

    def __call__(self):
        return torch.cat([
            self.core.agents.angvelocity[..., None]/360.,
            to_local_frame(self.core.agents.angles, self.core.agents.velocity)/10.], -1)

def random_empty_positions(geometries, n_agents, n_points):
    points = []
    for g in geometries:
        sample = np.stack((g.masks > 0).nonzero(), -1)

        # There might be fewer open points than we're asking for
        n_possible = min(len(sample)//n_agents, n_points)
        sample = sample[np.random.choice(np.arange(len(sample)), (n_possible, n_agents), replace=True)]

        # So repeat the sample until we've got enough
        sample = np.concatenate([sample]*int(n_points/len(sample)+1))[-n_points:]
        sample = np.random.permutation(sample)
        points.append(geometry.centers(sample, g.masks.shape, g.res).transpose(1, 0, 2))
    return arrdict.stack(points)
        
class RandomSpawns:

    def __init__(self, geometries, core, *args, n_spawns=100, **kwargs):
        self.core = core

        positions = random_empty_positions(geometries, core.n_agents, n_spawns)
        angles = core.random.uniform(-180, +180, (len(geometries), core.n_agents, n_spawns))
        self._spawns = arrdict.torchify(arrdict.arrdict(positions=positions, angles=angles)).to(core.device)

    def __call__(self, reset):
        core = self.core
        required = reset.nonzero(as_tuple=True)
        choices = torch.randint_like(required[0], 0, self._spawns.angles.shape[1])
        core.agents.angles[required] = self._spawns.angles[(*required, choices)] 
        core.agents.positions[required] = self._spawns.positions[(*required, choices)] 
        core.agents.velocity[required] = 0.
        core.agents.angvelocity[required] = 0.

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
        return arrdict.arrdict(length=self._lengths[d], max_length=self._max_lengths[d]).clone()
