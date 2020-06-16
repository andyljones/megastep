import numpy as np
import torch
from rebar import arrdict
from . import spaces

ACCEL = 5
ANG_ACCEL = 100
DECAY = .125

class SimpleMovement:

    def __init__(self):
        super().__init__()
        # noop, forward/backward, strafe left/right, turn left/right
        momenta = torch.tensor([[0., 0.], [0., 1.], [0.,-1.], [1., 0.], [-1.,0.], [0., 0.], [0., 0.]])
        angmomenta = torch.tensor([0., 0., 0., 0., 0., +1., -1.])
        self._actionset = arrdict(
            momenta=ACCEL/self.options.fps*momenta,
            angmomenta=ANG_ACCEL/self.options.fps*angmomenta
        ).to(self.device)

        self.action_space = arrdict(
            move=spaces.MultiDiscrete(self.options.n_agents, 7))

    def _to_global_frame(self, p):
        a = np.pi/180*self._agents.angles
        c, s = torch.cos(a), torch.sin(a)
        x, y = p[..., 0], p[..., 1]
        return torch.stack([c*x - s*y, s*x + c*y], -1)

    def _move(self, decisions):
        delta = self._actionset[decisions.actions.move]
        self._agents.angmomenta[:] = (1 - DECAY)*self._agents.angmomenta + delta.angmomenta
        self._agents.momenta[:] = (1 - DECAY)*self._agents.momenta + self._to_global_frame(delta.momenta)

class RGBObserver:

    def __init__(self):
        super().__init__()
        self.observation_space = arrdict(
            rgb=spaces.MultiImage(self.options.n_agents, 3, 1, self.options.res))

    def _downsample(self, screen):
        return screen.view(*screen.shape[:-1], screen.shape[-1]//self.options.supersample, self.options.supersample).mean(-1)

    def _observe(self):
        render = self._render()
        return arrdict(
            rgb=self._downsample(render.screen))
        