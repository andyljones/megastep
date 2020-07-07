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
    data = arrdict(
            angles=torch.zeros((n_envs, n_agents)),
            positions=torch.zeros((n_envs, n_agents, 2)),
            angmomenta=torch.zeros((n_envs, n_agents)),
            momenta=torch.zeros((n_envs, n_agents, 2)))
    return cuda.Agents(**data.to(device))

def select(x, d):
    if isinstance(x, dict):
        return x.__class__({k: select(v, d) for k, v in x.items()})
    if isinstance(x, torch.Tensor):
        return x[d]
    # Else it's a Ragged
    s = x.starts[d]
    e = s+x.widths[d]
    return x.vals[s:e]

class Core: 
    """The core rendering and physics interface"""

    def __init__(self, geometries, n_agents=1, res=64, supersample=8, fov=130, fps=10):
        self.geometries = geometries 
        self.n_envs = len(geometries)
        self.n_agents = n_agents
        self.res = res
        self.supersample = supersample
        self.fov = fov
        self.agent_radius = common.AGENT_RADIUS
        self.fps = fps
        self.random = np.random.RandomState(1)

        # TODO: This needs to be propagated to the C++ side
        self.device = torch.device('cuda')

        self.cuda = cuda()
        self.cuda.initialize(self.agent_radius, self.supersample*self.res, self.fov, self.fps)
        self.agents = init_agents(self.cuda, self.n_envs, self.n_agents, self.device)
        self.scene = scenery.init_scene(self.cuda, self.geometries, self.n_agents, self.device, self.random)
        self.progress = torch.ones((self.n_envs, self.n_agents), device=self.device)

        super().__init__()

    def state(self, d):
        scene = self.scene
        lines_s = scene.lines.starts[d]
        lines_e = scene.lines.starts[d]+scene.lines.widths[d]
        textures_s = scene.textures.starts[lines_s]
        textures_e = scene.textures.starts[lines_e-1] + scene.textures.widths[lines_e-1]
        textures = scene.textures.vals[textures_s:textures_e]
        baked = scene.baked.vals[textures_s:textures_e]

        options = ('n_envs', 'n_agents', 'res', 'supersample', 'fov', 'agent_radius', 'fps')
        options = {k: getattr(self, k) for k in options}

        return arrdict(
                    **options,
                    scene=arrdict(
                            frame=self.scene.frame,
                            lines=select(self.scene.lines, d),
                            lights=select(self.scene.lights, d),
                            start=textures_s,
                            widths=scene.textures.widths[lines_s:lines_e],
                            textures=textures,
                            baked=baked).clone(),
                    agents=arrdict(
                            angles=self.agents.angles[d], 
                            positions=self.agents.positions[d]).clone(),
                    progress=self.progress[d].clone())

    def env_full(self, x):
        """Returns a (n_env,) tensor on the device full of `obj`.
        
        This isn't strictly necessary, but you find yourself making these vectors so often it's useful sugar
        """
        dtypes = {bool: torch.bool, int: torch.int32, float: torch.float32}
        return torch.full((self.n_envs,), x, device=self.device, dtype=dtypes[type(x)])

    def agent_full(self, x):
        """Returns a (n_env,) tensor on the device full of `obj`.
        
        This isn't strictly necessary, but you find yourself making these vectors so often it's useful sugar
        """
        dtypes = {bool: torch.bool, int: torch.int32, float: torch.float32}
        return torch.full((self.n_envs, self.n_agents), x, device=self.device, dtype=dtypes[type(x)])