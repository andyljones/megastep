import numpy as np
from . import common, scenery, plotting
import torch
import logging
from rebar import dotdict, arrdict
from rebar.arrdict import tensorify, numpyify, cat

log = logging.getLogger(__name__)

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

def env_full_like(core, x):
    """Returns a (n_env,) tensor on the device full of `obj`.
    
    This isn't strictly necessary, but you find yourself making these vectors so often it's useful sugar
    """
    dtypes = {bool: torch.bool, int: torch.int32, float: torch.float32}
    return torch.full((core.n_envs,), x, device=core.device, dtype=dtypes[type(x)])

class Core: 

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

        self.cuda = common.cuda()
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
