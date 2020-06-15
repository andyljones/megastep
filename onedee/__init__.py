from .simulator import Simulator
from . import spaces
import torch
from rebar import arrdict
from functools import wraps

ACCEL = 5
ANG_ACCEL = 100
DECAY = .125

class Environment(Simulator):

    @wraps(Simulator.__init__)
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
        ).to(self.device)

    def _observe(self):
        render = self._render()
        return arrdict(
            rgb=self._downsample(render.screen))
    
    @torch.no_grad()
    def reset(self):
        reset = torch.ones((self.options.n_designs,), dtype=torch.bool, device=self.device)
        self._respawn(reset)
        return arrdict(
            obs=self._observe(), 
            reset=reset, 
            terminal=torch.zeros(self.options.n_designs, dtype=torch.bool, device=self.device), 
            reward=torch.zeros(self.options.n_designs, device=self.device))

    @torch.no_grad()
    def step(self, decisions):
        delta = self._actionset[decisions.actions.move]
        self._drones.angmomenta[:] = (1 - DECAY)*self._drones.angmomenta + delta.angmomenta
        self._drones.momenta[:] = (1 - DECAY)*self._drones.momenta + self._to_global_frame(delta.momenta)
        self._physics()
        return arrdict(
            obs=self._observe(), 
            reset=torch.zeros(self.options.n_designs, dtype=torch.bool, device=self.device), 
            terminal=torch.zeros(self.options.n_designs, dtype=torch.bool, device=self.device), 
            reward=torch.zeros(self.options.n_designs, device=self.device))

def example():
    import designs

    env = Environment([designs.box()])
    env.reset()
    forward = torch.tensor([[3]], device=env.device)
    for _ in range(10):
        env.step(arrdict(actions=arrdict(move=forward)))
    env.render()