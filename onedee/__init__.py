from .core import Core, env_full_like
from . import modules
import torch
from rebar import arrdict

class MinimalEnv:
    """A minimal environment with no rewards or resets, just to demonstrate physics and rendering"""

    def __init__(self, *args, **kwargs):
        self._core = Core(*args, **kwargs)
        self._mover = modules.SimpleMovement(self._core)
        self._observer = modules.RGBDObserver(self._core)
        self._respawner = modules.RandomSpawns(self._core)

    @torch.no_grad()
    def reset(self):
        self._respawner(env_full_like(self._core, True))
        return arrdict(obs=self._observer())

    @torch.no_grad()
    def step(self, decisions):
        self._mover(decisions)
        return arrdict(obs=self._observer())
    
class ExplorationEnv:

    def __init__(self, *args, max_length=512, **kwargs):
        self._core = Core(*args, **kwargs)
        self._mover = modules.SimpleMovement(self._core)
        self._observer = modules.RGBDObserver(self._core)
        self._respawner = modules.RandomSpawns(self._core)

        self._tex_to_env = self._core.scene.lines.inverse[self._core.scene.textures.inverse.to(torch.long)].to(torch.long)
        self._seen = torch.full_like(self._tex_to_env, False)

        self._length = env_full_like(self._core, 0)
        self._max_length = torch.randint_like(self._length, max_length//2, 2*max_length)

    def _tex_indices(self, aux): 
        scene = self._core.scene 
        mask = aux.indices >= 0
        result = torch.full_like(aux.indices, -1, dtype=torch.long)
        tex_n = (scene.lines.starts[:, None, None, None] + aux.indices)[mask]
        tex_w = scene.textures.widths[tex_n.to(torch.long)]
        tex_i = torch.min(torch.floor(tex_w.to(torch.float)*aux.locations[mask]), tex_w.to(torch.float)-1)
        tex_s = scene.textures.starts[tex_n.to(torch.long)]
        result[mask] = tex_s.to(torch.long) + tex_i.to(torch.long)
        return result.unsqueeze(2)

    def _reward(self, render, reset):
        texindices = self._tex_indices(render)
        seen = self._seen[texindices]
        self._seen[texindices] = True
        reward = (1 - seen.int()).reshape(seen.shape[0], -1).sum(-1)
        reward[reset] = 0
        reward = reward.float()/self.options.res
        return reward

    def _reset(self, reset):
        self._respawner(reset)
        self._seen[reset[self._tex_to_env]] = False
        self._length[reset] = 0

    @torch.no_grad()
    def reset(self):
        reset = self._full(True)
        self._reset(reset)
        render = self._observer.render()
        return arrdict(
            obs=self._observer(render), 
            reset=reset, 
            terminal=self._full(False), 
            reward=self._reward(render, reset))

    @torch.no_grad()
    def step(self, decisions):
        self._mover(decisions)
        self._length += 1

        reset = self._length == self._max_length
        self._reset(reset)
        render = self._observer.render()
        return arrdict(
            obs=self._observer(render), 
            reset=reset, 
            terminal=reset, 
            reward=self._reward(render, reset))
