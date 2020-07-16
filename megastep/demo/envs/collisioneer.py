import torch
from rebar import arrdict
from megastep import modules, core, plotting, toys, scene, cuda
import matplotlib.pyplot as plt

class Collisioneer:

    def __init__(self):
        geometries = 128*[toys.box()]
        scenery = scene.scenery(geometries, n_agents=1)
        self.core = core.Core(scenery)
        self.spawner = modules.RandomSpawns(geometries, self.core)
        self.depth = modules.Depth(self.core)
        self.movement = modules.SimpleMovement(self.core)

    def reset(self):
        self.spawner(self.core.agent_full(True))
        return arrdict.arrdict(obs=self.depth())

    def step(self, decision):
        # process decisions from the agent
        p = cuda.physics(self.c.scenery, self.c.agents)
        # post-collision alterations
        r = cuda.render(self.c.scenery, self.c.agents)
        # generate an observation and send it to the agent
        return world

