import numpy as np
import torch
from torch import nn
from onedee import spaces
from rebar import arrdict, stats, recurrence
from torch.nn import functional as F
from .transformer import Transformer

class Scaler(nn.Module):

    def __init__(self, width, com=10000):
        """Follows _Multi-task Deep Reinforcement Learning with PopArt_"""
        super().__init__()
        self._alpha = 1/(1+com)
        self.register_buffer('mu', torch.zeros(()))
        self.register_buffer('nu', torch.ones(()))
        self.layer = nn.Linear(width, 1)

    @property
    def sigma(self):
        return (self.nu - self.mu**2).pow(.5)

    def step(self, x):
        a = self._alpha
        mu = a*x.mean() + (1 - a)*self.mu
        nu = a*x.pow(2).mean() + (1 - a)*self.nu
        sigma = (nu - mu**2).pow(.5)
        
        self.layer.weight.data[:] = self.sigma/sigma*self.layer.weight
        self.layer.bias.data[:] = (self.sigma*self.layer.bias + self.mu - mu)/sigma

        self.mu[()] = mu
        self.nu[()] = nu

    def scale(self, x):
        return x/self.sigma

    def unscale(self, x):
        return x*self.sigma
    
    def norm(self, x):
        return (x - self.mu)/self.sigma
    
    def unnorm(self, x):
        return (x + self.mu)*self.sigma
    
    def forward(self, x):
        return self.layer(x)

class Agent(nn.Module):

    def __init__(self, observation_space, action_space, width=128):
        super().__init__()
        out = spaces.output(action_space, width)
        self.sampler = out.sample
        self.policy = recurrence.Sequential(
            spaces.intake(observation_space, width),
            Transformer(mem_len=64, d_model=width, n_layers=2, n_head=2),
            out)
        self.value = recurrence.Sequential(
            spaces.intake(observation_space, width),
            Transformer(mem_len=64, d_model=width, n_layers=2, n_head=2),
            spaces.ValueOutput(width, 1))

        self.register_buffer('v_mu', torch.tensor(0.))
        self.register_buffer('v_sigma', torch.tensor(1.))

    def forward(self, world, sample=False, value=False):
        outputs = arrdict(
            logits=self.policy(world.obs, reset=world.reset))
        if sample:
            outputs['actions'] = self.sampler(outputs.logits)
        if value:
            outputs['value_z'] = self.value(world.obs, reset=world.reset).squeeze(-1)
            outputs['value'] = self.v_mu + self.v_sigma * outputs.value_z
        return outputs

