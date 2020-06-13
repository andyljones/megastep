import gym
import torch
from torch import nn

class FlatIntake(nn.Module):

    def __init__(self, space, width):
        super().__init__()
        shape = space.shape if isinstance(space, gym.spaces.Space) else space
        self.core = nn.Sequential(
            nn.Linear(shape[0], width), nn.ReLU(),
            nn.Linear(width, width))

    def forward(self, obs):
        return self.core(obs)

class ImageIntake(nn.Module):

    def __init__(self, space, width):
        super().__init__()
        self.core = nn.Sequential(
                        nn.Conv2d(space.shape[0], 16, 8, stride=4), nn.ReLU(),
                        nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
                        nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())
        self.proj = nn.Sequential(
                        nn.Linear(32*7*7, width), nn.ReLU())

    def forward(self, obs):
        T, B, H, W, C = obs.shape
        obs = obs.permute(0, 1, 4, 2, 3).contiguous()
        if obs.dtype == torch.uint8:
            obs = obs/255.
        x = self.core(obs.reshape(T*B, C, H, W)).reshape(T, B, -1)
        return self.proj(x)

def intake(space, width):
    shape = space.shape if isinstance(space, gym.spaces.Space) else space
    if len(shape) == 1:
        return FlatIntake(space, width)
    elif len(shape) == 3:
        return ImageIntake(space, width)
    raise ValueError(f'Can\'t handle {space}')