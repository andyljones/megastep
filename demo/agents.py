import gym
import numpy as np
import torch
from torch import nn
from onedee import spaces

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
        D, C, H, W = shape

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

    def __init__(self, children, width):
        super().__init__()

        self.core = nn.Linear(len(children)*width, width)
        self.children = nn.ModuleDict(children)

    def forward(self, x):
        ys = [self.children[k](x[k]) for k in self.children]
        return self.core(torch.cat(ys, -1))

def intake(space, width):
    if isinstance(space, dict):
        children = type(space)({k: intake(v, width) for k, v in space.items()})
        return ConcatIntake(children, width)
    elif isinstance(space, spaces.MultiImage):
        return MultiImageIntake(space.shape, width)
    raise ValueError(f'Can\'t handle {space}')

class DictOutput(nn.Module):

    def __init__(self, children, width):
        super().__init__()

        self.core = nn.Linear(width, width*len(children))
        self.children = nn.ModuleDict(children)

    def forward(self, x):
        ys = torch.chunk(self.core(x), len(self.children), -1)
        return type(x)({k: v(ys[i]) for i, (k, v) in enumerate(self.children.items())})
    
    def sample(self, l):
        return type(x)({k: v.sample(l[i]) for i, (k, v) in enumerate(self.children.items())})


class DiscreteOutput(nn.Module):

    def __init__(self, shape, width):
        super().__init__()
        self.core = nn.Linear(width, np.prod(shape))
        self.shape = shape
    
    def forward(self, x):
        y = self.core(x).reshape(*x.shape[:-1], *self.shape)
        return F.logit_softmax(y, -1)

    def sample(self, l):
        return torch.distributions.Categorical(logits=logits).sample()

def output(space, width):
    if isinstance(space, dict):
        children = type(space)({k: output(v, width) for k, v in space.items()})
        return DictOutput(children, width)
    if isinstance(space, spaces.MultiDiscrete):
        return DiscreteOutput(space.shape, width)
    raise ValueError(f'Can\'t handle {space}')

class Agent(nn.Module):

    def __init__(self, observation_space, action_space, width=128):
        super().__init__()
        self.intake = intake(observation_space, width)
        self.torso = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU())
        self.policy = output(action_space, width)
        self.value = nn.Linear(width, 1)

        self.register_buffer('gen', torch.tensor(0))

    def forward(self, reaction, sample=False, value=False):
        x = self.intake(reaction.obs)
        x = self.torso(x)

        outputs = arrdict.arrdict(
            gen=self.gen.new_full(x.shape[:2], self.gen),
            logits=self.policy(x))
        if sample:
            outputs['action'] = self.policy.sample(outputs.logits)
        if value:
            outputs['value'] = self.value(x).squeeze(-1)
        return outputs

