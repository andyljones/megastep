import numpy as np
import torch
from .. import modules, core, plotting, spaces
from rebar import arrdict, dotdict
import matplotlib.pyplot as plt

class ExplorerEnv: 

    def __init__(self, *args, **kwargs):
        self._core = core.Core(*args, **kwargs)
        self._mover = modules.MomentumMovement(self._core)
        self._rgbd = modules.RGBD(self._core)
        self._imu = modules.IMU(self._core)
        self._respawner = modules.RandomSpawns(self._core)

        self.action_space = self._mover.space
        self.observation_space = dotdict(
            **self._rgbd.space,
            imu=self._imu.space,)

        self._tex_to_env = self._core.scene.lines.inverse[self._core.scene.textures.inverse.to(torch.long)].to(torch.long)
        self._seen = torch.full_like(self._tex_to_env, False)

        self._length = core.env_full_like(self._core, 0)

        self._potential = core.env_full_like(self._core, 0.)

        self._base_length = 256

        self.device = self._core.device

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

    def _reward(self, texindices, reset):
        self._seen[texindices] = True

        potential = torch.zeros_like(self._potential)
        potential.scatter_add_(0, self._tex_to_env, self._seen.float())

        #TODO: How to make the collision penalty a potential?
        reward = (potential - self._potential)/self._core.res - .1*(1 - self._progress).sum(-1)
        self._potential = potential

        # Should I render twice so that the last reward is accurate?
        reward[reset] = 0.

        #TODO: Get rid of this and add a reward scaling mechanism to the learner
        return reward

    def _reset(self, reset):
        self._respawner(reset)
        self._seen[reset[self._tex_to_env]] = False
        self._length[reset] = 0
        self._potential[reset] = 0

    @torch.no_grad()
    def reset(self):
        reset = core.env_full_like(self._core, True)
        self._reset(reset)
        render = self._rgbd.render()
        texindices = self._tex_indices(render)
        return arrdict(
            obs=arrdict(
                **self._rgbd(render), 
                imu=self._imu()), 
            reset=reset, 
            terminal=torch.zeros_like(reset), 
            reward=self._reward(texindices, reset))

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)
        self._length += 1

        reset = (self._length >= self._potential + self._base_length)
        self._reset(reset)
        render = self._rgbd.render()
        texindices = self._tex_indices(render)
        return arrdict(
            obs=arrdict(
                **self._rgbd(render), 
                imu=self._imu()), 
            reset=reset, 
            terminal=torch.zeros_like(reset), 
            reward=self._reward(texindices, reset))

    def state(self, d=0):
        seen = self._seen[self._tex_to_env == d]
        return arrdict(
            **self._core.state(d),
            obs=self._rgbd.state(d),
            potential=self._potential[d].clone(),
            seen=seen.clone(),
            length=self._length[d].clone(),
            max_length=self._potential[d].add(self._base_length).clone())

    @classmethod
    def plot_state(cls, state, zoom=False):
        fig = plt.figure()
        gs = plt.GridSpec(2, 2, fig, 0, 0, 1, 1)

        alpha = .1 + .9*state.seen.astype(float)
        # modifying this in place will bite me eventually. o for a lens
        state['scene']['textures'] = np.concatenate([state.scene.textures, alpha[:, None]], 1)
        ax = plotting.plot_core(state, plt.subplot(gs[:, 0]), zoom=zoom)
        plotting.plot_images(state.obs, [plt.subplot(gs[0, 1])])

        s = (f'length: {state.length:d}/{state.max_length:.0f}\n'
            f'potential: {state.potential:.0f}')
        ax.annotate(s, (5., 5.), xycoords='axes points')

        return fig

    def display(self, d=0):
        return self.plot_state(arrdict.numpyify(self.state(d)))

