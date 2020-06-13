import gym
import torch
from torch import nn

class FlatIntake(nn.Module):

    def __init__(self, shape, width):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(shape[0], width), nn.ReLU(),
            nn.Linear(width, width))

    def forward(self, obs):
        return self.core(obs)

class MultiImageIntake(nn.Module):

    def __init__(self, shape, width):
        super().__init__()
        D, C, H, W = space.shape

        self.conv = nn.Sequential(
                        nn.Conv2d(C, 16, (1, 8), stride=(1, 4)), nn.ReLU(),
                        nn.Conv2d(16, 32, (1, 4), stride=(1, 2)), nn.ReLU(),
                        nn.Conv2d(32, 32, (1, 3), stride=(1, 2)), nn.ReLU())

        zeros = torch.zeros((D, C, H, W))
        convwidth = self.conv(zeros).nelement()

        self.proj = nn.Sequential(
                        nn.Linear(convwidth, width), nn.ReLU())

    def forward(self, obs):
        T, B, D, C, H, W = obs.shape
        if obs.dtype == torch.uint8:
            obs = obs/255.
        x = self.core(obs.reshape(T*B*D, C, H, W)).reshape(T, B, -1)
        return self.proj(x)

class ConcatIntake(nn.Module):

    def __init__(self, subintakes, width):
        super().__init__()

        self.core = nn.Linear(len(subintakes)*width, width)
        self.subintakes = subintakes

    def forward(self, x):
        ys = [self.subintakes[k](x[k]) for k in self.subintakes]
        return self.core(torch.cat(ys, -1))

def intake(space, width):
    if isinstance(space, dict):
        subintakes = type(space)({k: intake(v, width) for k, v in space.items()})
        return ConcatIntake(subintakes, width)
    elif isinstance(space, gym.spaces.Box):
        if len(space.shape) == 1:
            return FlatIntake(space.shape, width)
        elif len(space.shape) == 4:
            return MultiImageIntake(space.shape, width)
    raise ValueError(f'Can\'t handle {space}')