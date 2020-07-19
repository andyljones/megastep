"""
TODO-DOCS Heads docs
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class MultiVectorIntake(nn.Module):

    def __init__(self, space, width):
        super().__init__()
        A, C = space.shape

        self.core = nn.Sequential(
                        nn.Linear(C, width), nn.ReLU(),)
        self.proj = nn.Sequential(
                        nn.Linear(A*width, width), nn.ReLU(),)
        
    def forward(self, obs, **kwargs):
        if obs.ndim == 3:
            return self.forward(obs[None], **kwargs).squeeze(0)

        T, B, A, C = obs.shape
        x = self.core(obs.reshape(T*B*A, C)).reshape(T, B, -1)
        return self.proj(x)

class MultiImageIntake(nn.Module):

    def __init__(self, space, width):
        super().__init__()
        A, C, H, W = space.shape

        self.conv = nn.Sequential(
                        nn.Conv2d(C, 32, (1, 8), stride=(1, 4)), nn.ReLU(),
                        nn.Conv2d(32, 64, (1, 4), stride=(1, 2)), nn.ReLU(),
                        nn.Conv2d(64, 128, (1, 3), stride=(1, 2)), nn.ReLU())

        zeros = torch.zeros((A, C, H, W))
        convwidth = self.conv(zeros).nelement()

        self.proj = nn.Sequential(
                        nn.Linear(convwidth, width), nn.ReLU(),
                        nn.Linear(width, width), nn.ReLU())

    def forward(self, obs, **kwargs):
        if obs.ndim == 5:
            return self.forward(obs[None], **kwargs).squeeze(0)

        T, B, A, C, H, W = obs.shape
        if obs.dtype == torch.uint8:
            obs = obs/255.
        x = self.conv(obs.reshape(T*B*A, C, H, W)).reshape(T, B, -1)
        return self.proj(x)

class ConcatIntake(nn.Module):

    def __init__(self, space, width):
        super().__init__()

        intakes = type(space)({k: intake(v, width) for k, v in space.items()})
        self.core = nn.Linear(len(intakes)*width, width)
        self.intakes = nn.ModuleDict(intakes)

    def forward(self, x, **kwargs):
        ys = [self.intakes[k](x[k]) for k in self.intakes]
        return self.core(torch.cat(ys, -1))

def intake(space, width):
    if isinstance(space, dict):
        return ConcatIntake(space, width)
    name = f'{type(space).__name__}Intake'
    if name in globals():
        return globals()[name](space, width)
    raise ValueError(f'Can\'t handle {space}')

class MultiDiscreteOutput(nn.Module):

    def __init__(self, space, width):
        super().__init__()
        shape = space.shape
        self.core = nn.Linear(width, int(np.prod(shape)))
        self.shape = shape
    
    def forward(self, x, **kwargs):
        y = self.core(x).reshape(*x.shape[:-1], *self.shape)
        return F.log_softmax(y, -1)

    def sample(self, logits, test=False):
        if test:
            return logits.argmax(-1)
        else:
            return torch.distributions.Categorical(logits=logits).sample()

class DictOutput(nn.Module):

    def __init__(self, space, width):
        super().__init__()
        self.core = nn.Linear(width, width*len(space))

        self._dtype = type(space)
        self.outputs = nn.ModuleDict({k: output(v, width) for k, v in space.items()})

    def forward(self, x, **kwargs):
        ys = torch.chunk(self.core(x), len(self.outputs), -1)
        return self._dtype({k: v(ys[i]) for i, (k, v) in enumerate(self.outputs.items())})
    
    def sample(self, l):
        return self._dtype({k: v.sample(l[k]) for k, v in self.outputs.items()})

class ValueOutput(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.core = nn.Linear(width, 1)

    def forward(self, x, **kwargs):
        return self.core.forward(x).squeeze(-1)

def output(space, width):
    if isinstance(space, dict):
        return DictOutput(space, width)
    name = f'{type(space).__name__}Output'
    if name in globals():
        return globals()[name](space, width)
    raise ValueError(f'Can\'t handle {space}')

