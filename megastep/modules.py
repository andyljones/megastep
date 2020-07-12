import numpy as np
import torch
from rebar import arrdict
from . import spaces, geometry, cuda

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
        self._core = core
        momenta = torch.tensor([[0., 0.], [0., 1.], [0.,-1.], [1., 0.], [-1.,0.], [0., 0.], [0., 0.]])
        angmomenta = torch.tensor([0., 0., 0., 0., 0., +1., -1.])
        self._actionset = arrdict.arrdict(
            momenta=accel/core.fps*momenta,
            angmomenta=ang_accel/core.fps*angmomenta
        ).to(core.device)

        self.space = spaces.MultiDiscrete(n_agents or core.n_agents, 7)

    def __call__(self, decision):
        core = self._core
        delta = self._actionset[decision.actions]
        core.agents.angmomenta[:] = delta.angmomenta
        core.agents.momenta[:] = to_global_frame(core.agents.angles, delta.momenta)
        cuda.physics(core.scene, core.agents, core.progress)

class MomentumMovement:

    def __init__(self, core, *args, accel=5, ang_accel=180, decay=.125, n_agents=None, **kwargs):
        # noop, forward/backward, strafe left/right, turn left/right
        self._core = core
        momenta = torch.tensor([[0., 0.], [0., 1.], [0.,-1.], [1., 0.], [-1.,0.], [0., 0.], [0., 0.]])
        angmomenta = torch.tensor([0., 0., 0., 0., 0., +1., -1.])
        self._actionset = arrdict.arrdict(
            momenta=accel/core.fps*momenta,
            angmomenta=ang_accel/core.fps*angmomenta
        ).to(core.device)

        self._decay = decay

        self.space = spaces.MultiDiscrete(n_agents or core.n_agents, 7)

    def __call__(self, decision):
        core = self._core
        delta = self._actionset[decision.actions]
        core.agents.angmomenta[:] = (1 - self._decay)*core.agents.angmomenta + delta.angmomenta
        core.agents.momenta[:] = (1 - self._decay)*core.agents.momenta + to_global_frame(core.agents.angles, delta.momenta)
        cuda.physics(core.scene, core.agents, core.progress)

def unpack(d):
    if isinstance(d, torch.Tensor):
        return d
    return arrdict.arrdict({k: unpack(getattr(d, k)) for k in dir(d) if not k.startswith('_')})
        
class RGBD:

    def __init__(self, core, *args, n_agents=None, max_depth=10, **kwargs):
        n_agents = n_agents or core.n_agents
        self._core = core
        self.space = arrdict.arrdict(
            rgb=spaces.MultiImage(n_agents, 3, 1, core.res),
            d=spaces.MultiImage(n_agents, 1, 1, core.res),)
        self._max_depth = max_depth

    def render(self):
        core = self._core
        render = unpack(cuda.render(core.scene, core.agents))
        render = arrdict.arrdict({k: v.unsqueeze(2) for k, v in render.items()})
        render['screen'] = render.screen.permute(0, 1, 4, 2, 3)
        return render

    def _downsample(self, screen):
        core = self._core
        return screen.view(*screen.shape[:-1], screen.shape[-1]//core.supersample, core.supersample).mean(-1)

    def __call__(self, render=None):
        render = self.render() if render is None else render
        depth = ((render.distances - self._core.agent_radius)/self._max_depth).clamp(0, 1)
        self._last_obs = arrdict.arrdict(
            rgb=self._downsample(render.screen),
            d=self._downsample(depth).unsqueeze(3))
        return self._last_obs
    
    def state(self, d):
        return self._last_obs[d].clone()

class IMU:

    def __init__(self, core, n_agents=None):
        self._core = core
        self.space = spaces.MultiVector(n_agents or core.n_agents, 3)

    def __call__(self):
        return torch.cat([
            self._core.agents.angmomenta[..., None]/360.,
            to_local_frame(self._core.agents.angles, self._core.agents.momenta)/10.], -1)

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
        points.append(geometry.centers(sample, g.masks.shape, g.res).transpose(1, 0, 2))
    return arrdict.stack(points)
        
class RandomSpawns:

    def __init__(self, core, *args, n_spawns=100, **kwargs):
        self._core = core

        positions = random_empty_positions(core, n_spawns)
        angles = core.random.uniform(-180, +180, (len(core.geometries), core.n_agents, n_spawns))
        self._spawns = arrdict.torchify(arrdict.arrdict(positions=positions, angles=angles)).to(core.device)

    def __call__(self, reset):
        core = self._core
        required = reset.nonzero(as_tuple=True)
        choices = torch.randint_like(required[0], 0, self._spawns.angles.shape[1])
        core.agents.angles[required] = self._spawns.angles[(*required, choices)] 
        core.agents.positions[required] = self._spawns.positions[(*required, choices)] 
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
        return arrdict.arrdict(length=self._lengths[d], max_length=self._max_lengths[d]).clone()
