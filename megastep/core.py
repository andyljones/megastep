import numpy as np
from . import scene, cuda, ragged
import torch
import logging
from rebar import arrdict, dotdict
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

AGENT_WIDTH = .15
TEXTURE_RES = .05

# Used for collision radius and near camera plane
AGENT_RADIUS = 1/2**.5*AGENT_WIDTH

def gamma_encode(x): 
    """Converts RGB data to `viewable values <https://en.wikipedia.org/wiki/Gamma_correction>`_."""
    return x**(1/2.2)

def gamma_decode(x):
    """Converts RGB data to `interpolatable values <https://en.wikipedia.org/wiki/Gamma_correction>`_."""
    return x**2.2

def _init_agents(n_envs, n_agents, device='cuda'):
    """Creates and returns an Agents datastructure"""
    data = arrdict.arrdict(
            angles=torch.zeros((n_envs, n_agents)),
            positions=torch.zeros((n_envs, n_agents, 2)),
            angvelocity=torch.zeros((n_envs, n_agents)),
            velocity=torch.zeros((n_envs, n_agents, 2)))
    return cuda.Agents(**data.to(device))

class Core: 

    def __init__(self, scenery, res=64, fov=130, fps=10):
        """The core rendering and physics interface. 

        To create the Core, you pass a :class:`~megastep.cuda.Scenery` that describes the environment. Once created, 
        the tensors hanging off of the Core give the state of the world, and that state can be advanced with the 
        functions in :mod:`~megastep.cuda`.

        :var agents: A :class:`~megastep.cuda.Agents` object describing the agents.
        :var scenery: A :class:`~megastep.cuda.Scenery` object describing the scene.
        :var progress: A (n_env, n_agent)-tensor giving how far the agent was able to move in the previous timestep as a 
            fraction of its intended movement, before running into an obstable. A value less than 1 means the agent collided
            with something. Useful for detecting collisions.
        :var n_envs: Number of environments. Same as the number of geometries passed in.
        :var n_agents: Number of agents.
        :var res: The horizontal resolution of observations. 
        :var fov: The field of view in degrees.
        :var agent_radius: The radius of a disc containing the agent, in meters.
        :var fps: The framerate.
        :var random: The seeded :class:`numpy.random.RandomState` used to initialize the environment. By reusing this
            in any extra random decisions made when generating the environments, it can be guaranteed you'll get the same
            environments every time.

        :param scenery: Describes the static parts of the environment.
        :type scenery: :class:`~megastep.cuda.Scenery`.
        :param n_agents: the number of agents to put in each environment. Defaults to 1.
        :type n_agents: int
        :param res: The horizontal resolution of the observations. The resolution must be less than 1024, as
            that's the `maximum number of CUDA threads in a block
            <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities>`_. Defaults to 64
            pixels.
        :type res: int
        :param fov: The field of view in degrees. Must be less than 180° due to how frames are rendered. Defaults to 130°.
        :type fov float:
        :param fps: The simulation frame rate/step rate. Defaults to 10. 
        :type fps: int
        """

        self.n_envs = len(scenery.lines.widths)
        self.n_agents = scenery.n_agents
        self.res = res
        self.fov = fov
        # TODO: Make this adjustable
        self.agent_radius = AGENT_RADIUS
        self.fps = fps
        self.random = np.random.RandomState(1)

        # TODO: This needs to be propagated to the C++ side
        self.device = scenery.model.device

        assert fov < 180, 'FOV should be less than 180°'

        cuda.initialize(self.agent_radius, self.res, self.fov, self.fps)
        self.scenery = scenery 
        self.agents = _init_agents(self.n_envs, self.n_agents, self.device)
        self.progress = torch.ones((self.n_envs, self.n_agents), device=self.device)

    def state(self, e):
        """Returns a :class:`~rebar.dotdict.dotdict` tree representing the state of environment ``e``.

        A typical state looks like this::

            arrdict:
            n_envs          1
            n_agents        4
            res             512
            fov             60
            agent_radius    0.10606601717798211
            fps             10
            scenery           arrdict:
                            model       Tensor((8, 2, 2), torch.float32)
                            lines       Tensor((307, 2, 2), torch.float32)
                            lights      Tensor((21, 3), torch.float32)
                            textures    <megastepcuda.Ragged2D object at 0x7fba34112eb0>
                            baked       <megastepcuda.Ragged1D object at 0x7fba34112670>
            agents          arrdict:
                            angles       Tensor((4,), torch.float32)
                            positions    Tensor((4, 2), torch.float32)
            progress        Tensor((4,), torch.float32)

        This state tree is usually passed onto a :ref:`plotting` function.""" 
        options = ('n_envs', 'n_agents', 'res', 'fov', 'agent_radius', 'fps')
        options = {k: getattr(self, k) for k in options}

        return arrdict.clone(dotdict.dotdict(
                **options,
                scenery=self.scenery.state(e),
                agents=self.agents.state(e),
                progress=self.progress[e]))
    
    @classmethod
    def plot_state(cls, state, ax=None, zoom=False):
        from . import plotting
        ax = ax or plt.axes()
        plotting.plot_lines(ax, state, zoom=zoom)
        plotting.plot_lights(ax,state)
        plotting.adjust_view(ax, state, zoom=zoom)
        plotting.plot_fov(ax, state)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def env_full(self, x):
        """Returns a (n_envs,)-tensor on the environment's device full of `x`.

        This isn't strictly required by the Core, but you find yourself making these vectors so often it's useful sugar.
        """
        dtypes = {bool: torch.bool, int: torch.int32, float: torch.float32}
        return torch.full((self.n_envs,), x, device=self.device, dtype=dtypes[type(x)])

    def agent_full(self, x):
        """Returns a (n_envs, n_agents)-tensor on the environment's device full of `x`.
        
        This isn't strictly required by the Core, but you find yourself making these vectors so often it's useful sugar.
        """
        dtypes = {bool: torch.bool, int: torch.int32, float: torch.float32}
        return torch.full((self.n_envs, self.n_agents), x, device=self.device, dtype=dtypes[type(x)])