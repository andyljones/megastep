from .simulator import Simulator
import torch
import gym
from rebar import arrdict

class Environment(Simulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = arrdict(
            rgb=gym.spaces.Box(0, 1, (self.options.n_drones, 3, 1, self.options.res)))
        self.action_space = arrdict(
            move=gym.spaces.MultiDiscrete((7,)*self.options.n_drones))

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
    def step(self, aug=None, actions=None):
        actions = torch.zeros((self.options.n_designs, self.options.n_drones), dtype=torch.int, device='cuda') if actions is None else actions
        self._act(arrdict(movement=arrdict(general=actions)))

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