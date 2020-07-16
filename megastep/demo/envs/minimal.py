from rebar import arrdict
from megastep import modules, core, toys, scene

class Minimal:

    def __init__(self):
        """A minimal environment, with a box env, depth observations and simple movement. A good foundation for
        building your own environments. See :ref:`the simple environment tutorial for details <simple-env>`"""

        geometries = [toys.box()]
        scenery = scene.scenery(geometries, n_agents=1)
        self.core = core.Core(scenery)
        self.spawner = modules.RandomSpawns(geometries, self.core)
        self.depth = modules.Depth(self.core)
        self.movement = modules.SimpleMovement(self.core)

        self.obs_space = self.depth.space
        self.action_space = self.movement.space

    def reset(self):
        self.spawner(self.core.agent_full(True))
        return arrdict.arrdict(obs=self.depth())

    def step(self, decision):
        self.movement(decision)
        return arrdict.arrdict(obs=self.depth())

