from .core import Core
from . import modules
import torch
from rebar import arrdict
from functools import wraps

class MinimalEnv(modules.SimpleMovement, modules.RGBObserver, modules.RandomSpawns):
    """A minimal environment with no rewards or resets, just to demonstrate physics and rendering"""

    @wraps(Core.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def reset(self):
        self._respawn(self._full(True))
        return arrdict(obs=self._observe())

    @torch.no_grad()
    def step(self, decisions):
        self._move(decisions)
        return arrdict(obs=self._observe())
    
class ExplorationEnv(modules.SimpleMovement, modules.RandomSpawns, modules.RGBDObserver):

    @wraps(Core.__init__)
    def __init__(self, *args, max_length=512, **kwargs):
        super().__init__(*args, **kwargs)
        self._tex_to_env = self._scene.lines.inverse[self._scene.textures.inverse.to(torch.long)].to(torch.long)
        self._seen = torch.full_like(self._tex_to_env, False)

        self._length = self._full(0)
        self._max_length = torch.randint_like(self._length, max_length//2, 2*max_length)

    def _tex_indices(self, aux): 
        mask = aux.indices >= 0
        result = torch.full_like(aux.indices, -1, dtype=torch.long)
        tex_n = (self._scene.lines.starts[:, None, None, None] + aux.indices)[mask]
        tex_w = self._scene.textures.widths[tex_n.to(torch.long)]
        tex_i = torch.min(torch.floor(tex_w.to(torch.float)*aux.locations[mask]), tex_w.to(torch.float)-1)
        tex_s = self._scene.textures.starts[tex_n.to(torch.long)]
        result[mask] = tex_s.to(torch.long) + tex_i.to(torch.long)
        return result.unsqueeze(2)

    def _render(self):
        render = modules.unpack(self._cuda.render(self._agents, self._scene))
        render = arrdict({k: v.unsqueeze(2) for k, v in render.items()})
        render['screen'] = render.screen.permute(0, 1, 4, 2, 3)
        render['texindices'] = self._tex_indices(render)
        return render

    def _reward(self, render, reset):
        seen = self._seen[render.texindices]
        self._seen[render.texindices] = True
        reward = (1 - seen.int()).reshape(seen.shape[0], -1).sum(-1)
        reward[reset] = 0
        reward = reward.float()/self.options.res
        return reward

    def _reset(self, reset):
        self._respawn(reset)
        self._seen[reset[self._tex_to_env]] = False
        self._length[reset] = 0

    @torch.no_grad()
    def reset(self):
        reset = self._full(True)
        self._reset(reset)
        render = self._render()
        return arrdict(
            obs=self._observe(render), 
            reset=reset, 
            terminal=self._full(False), 
            )

    @torch.no_grad()
    def step(self, decisions):
        self._move(decisions)
        self._length += 1

        reset = self._length == self._max_length
        self._respawn(reset)
        render = self._render()
        return arrdict(
            obs=self._observe(render), 
            reset=reset, 
            terminal=reset, 
            )
