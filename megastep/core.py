import numpy as np
from . import scenery
import torch
import logging
from rebar import arrdict
import torch.utils.cpp_extension
from pkg_resources import resource_filename
import sysconfig

log = logging.getLogger(__name__)

AGENT_WIDTH = .15
TEXTURE_RES = .05

# Used for collision radius and near camera plane
AGENT_RADIUS = 1/2**.5*AGENT_WIDTH

DEBUG = False

def cuda():
    """Compiles and loads the C++ side of onedee, returning it as a Python module.
    
    The best explanation of what's going on here is the `PyTorch C++ extension tutorial
    <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_ .
    
    I have very limited experience with distributing binaries, so while I've _tried_ to reference the library paths
    in a platform-independent way, there is a good chance they'll turn out to be dependent after all. Sorry. Submit
    an issue and explain a better way to me!
    
    The libraries listed are - I believe - the minimal possible to allow onedee's compilation. The default library
    set for PyTorch extensions is much larger and slower to compile.
    """
    [torch_libdir] = torch.utils.cpp_extension.library_paths()
    python_libdir = sysconfig.get_config_var('LIBDIR')
    libpython_ver = sysconfig.get_config_var('LDVERSION')
    return torch.utils.cpp_extension.load(
        name='onedeekernels', 
        sources=[resource_filename(__package__, f'src/{fn}') for fn in ('wrappers.cpp', 'kernels.cu')], 
        extra_cflags=['-std=c++17'] + (['-g'] if DEBUG else []), 
        extra_cuda_cflags=['--use_fast_math', '-lineinfo', '-std=c++14'] + (['-g', '-G'] if DEBUG else []),
        extra_ldflags=[
            f'-lpython{libpython_ver}', '-ltorch', '-ltorch_python', '-lc10_cuda', '-lc10', 
            f'-L{torch_libdir}', f'-Wl,-rpath,{torch_libdir}',
            f'-L{python_libdir}', f'-Wl,-rpath,{python_libdir}'])

def gamma_encode(x): 
    """Converts to viewable values"""
    return x**(1/2.2)

def gamma_decode(x):
    """Converts to interpolatable values"""
    return x**2.2

def init_agents(cuda, n_envs, n_agents, device='cuda'):
    """Creates and returns an Agents datastructure"""
    data = arrdict.arrdict(
            angles=torch.zeros((n_envs, n_agents)),
            positions=torch.zeros((n_envs, n_agents, 2)),
            angmomenta=torch.zeros((n_envs, n_agents)),
            momenta=torch.zeros((n_envs, n_agents, 2)))
    return cuda.Agents(**data.to(device))

class Core: 
    """The core rendering and physics interface. 

    To create the Core, you pass a collection of :ref:`geometries <geometry>` that describe the
    static environment. Once created, the tensors hanging off of the Core give the state of the world,
    and that state can be advanced with the functions hanging off of ``.cuda`` .

    :param geometries: A list-like of :ref:`geometries <geometry>` that describe each static environment.
    :param n_agents: the number of agents to put in each environment. Defaults to 1.
    :type n_agents: int
    :param res: The horizontal resolution of the observations. The resolution times the supersampling factor must be
        less than 1024, as that's the `maximum number of CUDA threads in a block
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities>`_. Defaults to 64 pixels.
    :type res: int
    :param supersample: The multiplier at which to render the observations. A higher value gives better antialiasing,
        but makes for a slower simulation. The resolution times the supersampling factor must be less than 1024, as
        that's the `maximum number of CUDA threads in a block
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities>`_. Defaults to 64 pixels.
        Defaults to 8.
    :type supersample: int
    :param fov: The field of view in degrees. Defaults to 130°.
    :type fov float:
    :param fps: The simulation frame rate/step rate. Defaults to 10. 
    :type fps: int
    
    """

    def __init__(self, geometries, n_agents=1, res=64, supersample=8, fov=130, fps=10):
        self.geometries = list(geometries)
        self.n_envs = len(geometries)
        self.n_agents = n_agents
        self.res = res
        self.supersample = supersample
        self.fov = fov
        self.agent_radius = AGENT_RADIUS
        self.fps = fps
        self.random = np.random.RandomState(1)

        # TODO: This needs to be propagated to the C++ side
        self.device = torch.device('cuda')

        self.cuda = cuda()
        self.cuda.initialize(self.agent_radius, self.supersample*self.res, self.fov, self.fps)
        self.agents = init_agents(self.cuda, self.n_envs, self.n_agents, self.device)
        self.scene = scenery.init_scene(self.cuda, self.geometries, self.n_agents, self.device, self.random)
        self.progress = torch.ones((self.n_envs, self.n_agents), device=self.device)

    def state(self, d):
        scene = self.scene
        sd, ed = scene.lines.starts[d], scene.lines.ends[d]

        options = ('n_envs', 'n_agents', 'res', 'supersample', 'fov', 'agent_radius', 'fps')
        options = {k: getattr(self, k) for k in options}

        return arrdict.arrdict(
                    **options,
                    scene=arrdict.arrdict(
                            frame=self.scene.frame,
                            lines=self.scene.lines[d],
                            lights=self.scene.lights[d],
                            #TODO: Fix up ragged so this works
                            textures=self.scene.textures[sd:ed],
                            baked=self.scene.baked[sd:ed]).clone(),
                    agents=arrdict.arrdict(
                            angles=self.agents.angles[d], 
                            positions=self.agents.positions[d]).clone(),
                    progress=self.progress[d].clone())

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