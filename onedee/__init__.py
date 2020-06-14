from .simulator import Simulator
from . import spaces
import torch
from rebar import arrdict

ACCEL = 5
ANG_ACCEL = 100
DECAY = .125

class Environment(Simulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = arrdict(
            rgb=spaces.MultiImage(self.options.n_drones, 3, 1, self.options.res))
        self.action_space = arrdict(
            move=spaces.MultiDiscrete(self.options.n_drones, 7))
            
        # noop, forward/backward, strafe left/right, turn left/right
        momenta = torch.tensor([[0., 0.], [0., 1.], [0.,-1.], [1., 0.], [-1.,0.], [0., 0.], [0., 0.]])
        angmomenta = torch.tensor([0., 0., 0., 0., 0., +1., -1.])
        self._actionset = arrdict(
            momenta=ACCEL/self.options.fps*momenta,
            angmomenta=ANG_ACCEL/self.options.fps*angmomenta
        ).cuda()

    def _observe(self):
        render = self._render()
        return arrdict(
            rgb=self._downsample(render.screen))
    
    @torch.no_grad()
    def reset(self):
        reset = torch.ones((self.options.n_designs,), dtype=torch.bool, device='cuda')
        self._respawn(reset)
        return arrdict(obs=self._observe())

    @torch.no_grad()
    def step(self, decisions):
        delta = self._actionset[decisions.actions.move]
        self._drones.angmomenta[:] = (1 - DECAY)*self._drones.angmomenta + delta.angmomenta
        self._drones.momenta[:] = (1 - DECAY)*self._drones.momenta + self._to_global_frame(delta.momenta)
        self._physics()

        reset = torch.zeros(self.options.n_designs, dtype=torch.bool, device='cuda')
        terminal = torch.zeros(self.options.n_designs, dtype=torch.bool, device='cuda')

        obs = self._observe()
        reward = torch.zeros(self.options.n_designs, device='cuda')
        return arrdict(reset=reset, terminal=terminal, obs=obs, reward=reward)

def example():
    from drones.designs.toy import collision_test

    forward = torch.tensor([[3]], dtype=torch.int, device='cuda')
    env = Environment(designer=lambda r: collision_test(), n_designs=1)
    env.reset()
    for _ in range(10):
        env.step(actions=forward)
    env.render()