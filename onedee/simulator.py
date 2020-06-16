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
    'fov': 130, # widest FOV among common FPV agents
    'radius': common.AGENT_RADIUS,
    'max_depth': 10,
    'fps': 10,
}

def init_respawns(cuda, geometries, n_agents, device='cuda', random=np.random):
    assert n_agents == 1

    respawns = []
    for g in geometries:
        sample = np.stack((g.masks == 0).nonzero(), -1)
        sample = sample[random.choice(np.arange(len(sample)), 100)]
        
        i, j = sample.T + .5
        xy = g.res*np.stack([j, g.masks.shape[0] - i], -1)

        respawns.append(arrdict({
            'centers': xy[:, None],
            'widths': len(xy),
            'radii': np.zeros((len(xy), 1)),
            'lowers': np.zeros((len(xy), 1)),
            'uppers': np.zeros((len(xy), 1))}))
    respawns = tensorify(cat(respawns)).to(device)
    return cuda.Respawns(**respawns)

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

class Simulator: 

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
        self._respawns = init_respawns(self._cuda, self._geometries, self.options.n_agents, self.device, self.options.random)
        self._agents = init_agents(self._cuda, self.options.n_envs, self.options.n_agents, self.device)
        self._scene = scenery.init_scene(self._cuda, self._geometries, self.options.n_agents, self.device, self.options.random)
 
        # Defined here for easy overriding in subclasses
        self._plot = plotting.plot

    @property
    def device(self):
        """The device that the sim sits on. For now this is fixed to the default 'cuda' device"""
        return self._device

    def _to_global_frame(self, p):
        a = np.pi/180*self._agents.angles
        c, s = torch.cos(a), torch.sin(a)
        x, y = p[..., 0], p[..., 1]
        return torch.stack([c*x - s*y, s*x + c*y], -1)

    def _physics(self):
        self._cuda.physics(self._scene, self._agents)

    def _respawn(self, reset):
        self._cuda.respawn(reset, self._respawns, self._agents)

    def _downsample(self, screen, agg='mean'):
        view = screen.view(*screen.shape[:-1], screen.shape[-1]//self.options.supersample, self.options.supersample)
        if agg == 'mean':
            return view.mean(-1)
        elif agg == 'min':
            return view.min(-1).values
        elif agg == 'first':
            return view[..., 0]

    def _upsample(self, screen):
        return screen.unsqueeze(-1).repeat(1, 1, 1, 1, 1, self.options.supersample).view(*screen.shape[:-1], screen.shape[-1]*self.options.supersample)

    def _compress(self, distances):
        return (1 - distances/self.options.max_depth).clamp(0, 1) 

    def _render(self):
        render = common.unpack(self._cuda.render(self._agents, self._scene))
        render = arrdict({k: v.unsqueeze(2) for k, v in render.items()})
        render['screen'] = render.screen.permute(0, 1, 4, 2, 3)
        return render

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