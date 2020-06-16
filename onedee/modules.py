import numpy as np
import torch
from rebar import arrdict
from rebar.arrdict import cat, stack, tensorify
from . import spaces, core

class SimpleMovement(core.Core):

    def __init__(self, *args, accel=5, ang_accel=100, decay=.125, **kwargs):
        super().__init__(*args, **kwargs)
        # noop, forward/backward, strafe left/right, turn left/right
        momenta = torch.tensor([[0., 0.], [0., 1.], [0.,-1.], [1., 0.], [-1.,0.], [0., 0.], [0., 0.]])
        angmomenta = torch.tensor([0., 0., 0., 0., 0., +1., -1.])
        self._actionset = arrdict(
            momenta=accel/self.options.fps*momenta,
            angmomenta=ang_accel/self.options.fps*angmomenta
        ).to(self.device)

        self.options.decay = decay

        self.action_space = arrdict(
            move=spaces.MultiDiscrete(self.options.n_agents, 7))

    def _to_global_frame(self, p):
        a = np.pi/180*self._agents.angles
        c, s = torch.cos(a), torch.sin(a)
        x, y = p[..., 0], p[..., 1]
        return torch.stack([c*x - s*y, s*x + c*y], -1)

    def _move(self, decisions):
        delta = self._actionset[decisions.actions.move]
        self._agents.angmomenta[:] = (1 - self.options.decay)*self._agents.angmomenta + delta.angmomenta
        self._agents.momenta[:] = (1 - self.options.decay)*self._agents.momenta + self._to_global_frame(delta.momenta)
        self._cuda.physics(self._scene, self._agents)

def unpack(d):
    if isinstance(d, torch.Tensor):
        return d
    return arrdict({k: unpack(getattr(d, k)) for k in dir(d) if not k.startswith('_')})

class RGBObserver(core.Core):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = arrdict(
            rgb=spaces.MultiImage(self.options.n_agents, 3, 1, self.options.res))

    def _downsample(self, screen):
        return screen.view(*screen.shape[:-1], screen.shape[-1]//self.options.supersample, self.options.supersample).mean(-1)

    def _observe(self, render=None):
        if render is None:
            render = unpack(self._cuda.render(self._agents, self._scene))
            render = arrdict({k: v.unsqueeze(2) for k, v in render.items()})
            render['screen'] = render.screen.permute(0, 1, 4, 2, 3)
        return arrdict(
            rgb=self._downsample(render.screen))
        

class RandomSpawns(core.Core):

    def __init__(self, *args, n_spawns=100, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.options.n_agents == 1
        self.options.n_spawns = n_spawns

        spawns = []
        for g in self._geometries:
            sample = np.stack((g.masks == 0).nonzero(), -1)
            sample = sample[self.options.random.choice(np.arange(len(sample)), n_spawns)]
            
            i, j = sample.T + .5
            xy = g.res*np.stack([j, g.masks.shape[0] - i], -1)

            spawns.append(arrdict({
                'positions': xy[:, None],
                'angles': self.options.random.uniform(-180, +180, (n_spawns, self.options.n_agents))}))

        self._spawns = tensorify(stack(spawns)).to(self.device)

    def _respawn(self, reset):
        required = reset.nonzero().squeeze(-1)
        choices = torch.randint_like(required, 0, self.options.n_spawns)
        self._agents.angles[required] = self._spawns.angles[required, choices] 
        self._agents.positions[required] = self._spawns.positions[required, choices] 
        self._agents.momenta[required] = 0.
        self._agents.angmomenta[required] = 0.

