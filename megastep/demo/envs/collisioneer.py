import torch
from rebar import arrdict
from megastep import modules, core, plotting, toys, scene, cuda
import matplotlib.pyplot as plt

class Collisioneer:

    def __init__(self):
        geometries = 128*[toys.box()]
        scenery = scene.scenery(geometries, n_agents=1)
        self.c = cuda.Core(scenery)

    def reset(self):
        r = cuda.render(self.c.scenery, self.c.agents)
        return world

    def step(self, decision):
        # process decisions from the agent
        p = cuda.physics(self.c.scenery, self.c.agents)
        # post-collision alterations
        r = cuda.render(self.c.scenery, self.c.agents)
        # generate an observation and send it to the agent
        return world

