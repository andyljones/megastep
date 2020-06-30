from .. import modules, core

class Tag:

    def __init__(self, *args, **kwargs):
        self._core = core.Core(*args, **kwargs)
        self._rgbd = modules.RGBD(self._core)
        self._mover = modules.MomentumMovement(self._core)
        self._respawner = modules.RandomSpawns(self._core)

    def reset(self):
        self._respawner(core.env_full(True))
        return arrdict(
            obs=self._observer(),
            reward=zeros(self.n_envs, self.device),
            reset=trues(self.n_envs, self.device),
            terminal=trues(self.n_envs, self.device),)
