import numpy as np
from . import common, scenery, plotting
import torch
import logging
from rebar import dotdict, arrdict
from rebar.arrdict import tensorify, numpyify, cat

log = logging.getLogger(__name__)

DEFAULTS = {
    'res': 64,
    'supersample': 8,
    'random': np.random.RandomState(12041955),
    'fov': 130, # widest FOV among common FPV drones
    'radius': common.AGENT_RADIUS,
    'fps': 10,
}

def init_agents(cuda, n_envs, n_agents, device='cuda'):
    data = arrdict(
            angles=tensorify(np.full((n_envs, n_agents), np.nan)),
            positions=tensorify(np.full((n_envs, n_agents, 2), np.nan)),
            angmomenta=tensorify(np.full((n_envs, n_agents), np.nan)),
            momenta=tensorify(np.full((n_envs, n_agents, 2), np.nan)))
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

    def __init__(self, geometries, n_agents=1, **kwargs):
        self._geometries = geometries 
        self.options = dotdict({
            **DEFAULTS, 
            **kwargs, 
            'n_agents': n_agents,
            'n_envs': len(self._geometries)})

        # TODO: This needs to be propagated to the C++ side
        self._device = torch.device('cuda')

        self._cuda = common.cuda(**self.options)
        self._cuda.initialize(self.options.radius, self.options.supersample*self.options.res, self.options.fov, self.options.fps)
        self._agents = init_agents(self._cuda, self.options.n_envs, self.options.n_agents, self.device)
        self._scene = scenery.init_scene(self._cuda, self._geometries, self.options.n_agents, self.device, self.options.random)
 
        # Defined here for easy overriding in subclasses
        self._plot = plotting.plot

        super().__init__()

    @property
    def device(self):
        """The device that the sim sits on. For now this is fixed to the default 'cuda' device"""
        return self._device

    def _full(self, obj):
        """Returns a (n_env,) tensor on the device full of `obj`.
        
        This isn't strictly necessary, but you find yourself making these vectors so often it's useful sugar
        """
        dtypes = {bool: torch.bool, int: torch.int32, float: torch.float32}
        return torch.full((self.options.n_envs,), obj, device=self.device, dtype=dtypes[type(obj)])

    def state(self, d):
        scene = self._scene
        lines_s = scene.lines.starts[d]
        lines_e = scene.lines.starts[d]+scene.lines.widths[d]
        textures_s = scene.textures.starts[lines_s]
        textures_e = scene.textures.starts[lines_e-1] + scene.textures.widths[lines_e-1]
        textures = scene.textures.vals[textures_s:textures_e]
        baked = scene.baked.vals[textures_s:textures_e]

        return arrdict(
                    options=arrdict({k: v for k, v in self.options.items() if k != 'random'}),
                    scene=arrdict(
                            frame=self._scene.frame,
                            lines=select(self._scene.lines, d),
                            lights=select(self._scene.lights, d),
                            start=textures_s,
                            widths=scene.textures.widths[lines_s:lines_e],
                            textures=textures,
                            baked=baked).clone(),
                    agents=arrdict(
                            angles=self._agents.angles[d], 
                            positions=self._agents.positions[d]).clone(),)

    def display(self, d=0):
        self._plot(numpyify(self.state(d)))

