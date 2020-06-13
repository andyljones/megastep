import numpy as np
import torch
from ..simulator import Simulator
from ...designs.common import Design
from rebar import dotdict

def mock_design(center):
    return Design(
        id='mock',
        centers=[[center]],
        radii=[[0.]],
        lights=[[.5, 2]],
        walls=[[[2, 1], [2, 3]]],
        lowers=[[0]],
        uppers=[[0]])

def run():
    sim = Simulator(lambda: mock_design([1., 2]), n_designs=1)
    sim.reset()

    nothing = dotdict(movement=dotdict(general=torch.tensor([[0]]).int().cuda()))
    forward = dotdict(movement=dotdict(general=torch.tensor([[3]]).int().cuda()))

    sim._drones.positions[:] = torch.tensor([[[1.85, 2]]])
    sim._drones.momenta[:] = torch.tensor([[[2., 0.]]])
    sim.step(nothing)

    torch.stack([sim._drones.positions[0, 0, 0], sim._drones.momenta[0, 0, 0]])
