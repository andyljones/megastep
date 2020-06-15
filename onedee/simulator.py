import numpy as np
from . import common, scenery, plotting
import torch
import logging
from rebar import dotdict, arrdict
from rebar.arrdict import tensorify, numpyify

log = logging.getLogger(__name__)

DEFAULTS = {
    'res': 64,
    'supersample': 8,
    'random': np.random.RandomState(12041955),
    'fov': 130, # widest FOV among common FPV drones
    'radius': common.DRONE_RADIUS,
    'max_depth': 10,
    'fps': 10,
}

def init_respawns(cuda, designs, device='cuda'):
    fields = ('centers', 'radii', 'lowers', 'uppers')
    data = arrdict({n: torch.cat([tensorify(getattr(d, n)) for d in designs]) for n in fields})
    data['widths'] = tensorify([d.respawns for d in designs]) 
    return cuda.Respawns(**data.to(device))

def init_drones(cuda, designs, device='cuda'):
    n_designs = len(designs)
    (n_drones,) = {d.n_drones for d in designs}
    data = arrdict(
            angles=tensorify(np.full((n_designs, n_drones), np.nan)),
            positions=tensorify(np.full((n_designs, n_drones, 2), np.nan)),
            angmomenta=tensorify(np.full((n_designs, n_drones), np.nan)),
            momenta=tensorify(np.full((n_designs, n_drones, 2), np.nan)))
    return cuda.Drones(**data.to(device)), n_drones

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

    @torch.no_grad()
    def __init__(self, designs, **kwargs):
        self._designs = designs 
        self.options = dotdict({**DEFAULTS, **kwargs, 'n_designs': len(self._designs)})

        # TODO: This needs to be propagated to the C++ side
        self._device = torch.device('cuda')

        self._cuda = common.cuda(**self.options)
        self._respawns = init_respawns(self._cuda, self._designs, self.device)
        self._drones, self.options['n_drones'] = init_drones(self._cuda, self._designs, self.device)
        self._scene = scenery.init_scene(self._cuda, self._designs, self.device, random=self.options.random)

        # Defined here for easy overriding in subclasses
        self._plot = plotting.plot

    @property
    def device(self):
        """The device that the sim sits on. For now this is fixed to the default 'cuda' device"""
        return self._device

    def _to_global_frame(self, p):
        a = np.pi/180*self._drones.angles
        c, s = torch.cos(a), torch.sin(a)
        x, y = p[..., 0], p[..., 1]
        return torch.stack([c*x - s*y, s*x + c*y], -1)

    def _physics(self):
        self._cuda.physics(self._scene, self._drones)

    def _respawn(self, reset):
        self._cuda.respawn(reset, self._respawns, self._drones)

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
        render = common.unpack(self._cuda.render(self._drones, self._scene))
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
                    drones=arrdict(
                            angles=self._drones.angles[d], 
                            positions=self._drones.positions[d]).clone(),)

    def display(self, d=0):
        return self._plot(numpyify(self.state(d)))