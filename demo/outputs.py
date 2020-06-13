import gym
import torch
from torch import nn

class DictOutput(nn.Module):

    def __init__(self, suboutputs, width):
        super().__init__()

        self.core = nn.Linear(width, width*len(suboutputs))
        self.suboutputs = suboutputs

    def forward(self, x):
        ys = torch.chunk(self.core(x), len(self.suboutputs), -1)
        return type(self.suboutputs)({k: ys[i] for i, k in enumerate(self.suboutputs)})

class DiscreteOutput(nn.Module):

    def __init__(self, shape, width):
        self.core = nn.Linear(width, sum(shape))
        self.shape = shape
    
    def forward(self, x):
        y = self.core(x)

        probs
        start = 0
        for n in self.shape:
            F.log_softmax(y[..., start:start+n])
            start += n


def outputs(space, width):
    if isinstance(space, dict):
        suboutputs = type(space)({k: outputs(v, width) for k, v in space.items()})
        return DictOutput(suboutputs, width)