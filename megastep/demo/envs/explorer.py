"""TODO-DOCS Explorer docs"""
import numpy as np
import torch
from megastep import modules, core, plotting, scene, cubicasa
from rebar import arrdict, dotdict
import matplotlib.pyplot as plt

class Explorer: 

    def __init__(self, n_envs, *args, **kwargs):
        geometries = cubicasa.sample(n_envs)
        scenery = scene.scenery(geometries, 1)
        self.core = core.Core(scenery, *args, res=4*64, fov=130, **kwargs)
        self._rgb = modules.RGB(self.core, n_agents=1, subsample=4)
        self._depth = modules.Depth(self.core, n_agents=1, subsample=4)
        self._mover = modules.MomentumMovement(self.core)
        self._imu = modules.IMU(self.core)
        self._respawner = modules.RandomSpawns(geometries, self.core)

        self.action_space = self._mover.space
        self.obs_space = dotdict.dotdict(
            rgb=self._rgb.space,
            d=self._depth.space,
            imu=self._imu.space)

        self._tex_to_env = self.core.scenery.lines.inverse[self.core.scenery.textures.inverse.long()].long()
        self._seen = torch.full_like(self._tex_to_env, False)
        self._potential = self.core.env_full(0.)

        self._lengths = torch.zeros(self.core.n_envs, device=self.core.device, dtype=torch.int)

        self.device = self.core.device

    def _tex_indices(self, aux): 
        scenery = self.core.scenery 
        mask = aux.indices >= 0
        result = torch.full_like(aux.indices, -1, dtype=torch.long)
        tex_n = (scenery.lines.starts[:, None, None, None] + aux.indices)[mask]
        tex_w = scenery.textures.widths[tex_n.to(torch.long)]
        tex_i = torch.min(torch.floor(tex_w.to(torch.float)*aux.locations[mask]), tex_w.to(torch.float)-1)
        tex_s = scenery.textures.starts[tex_n.to(torch.long)]
        result[mask] = tex_s.to(torch.long) + tex_i.to(torch.long)
        return result.unsqueeze(2)

    def _reward(self, r, reset):
        texindices = self._tex_indices(r)
        self._seen[texindices] = True

        potential = torch.zeros_like(self._potential)
        potential.scatter_add_(0, self._tex_to_env, self._seen.float())

        reward = (potential - self._potential)/(self.core.res//self._rgb.subsample)
        self._potential = potential

        # Should I render twice so that the last reward is accurate?
        reward[reset] = 0.

        return reward

    def _observe(self, reset):
        r = modules.render(self.core)
        obs = arrdict.arrdict(
                rgb=self._rgb(r), 
                d=self._depth(r), 
                imu=self._imu())
        reward = self._reward(r, reset)
        return obs, reward

    def _reset(self, reset=None):
        self._respawner(reset.unsqueeze(-1))
        self._seen[reset[self._tex_to_env]] = False
        self._potential[reset] = 0
        self._lengths[reset] = 0

    @torch.no_grad()
    def reset(self):
        reset = self.core.env_full(True)
        self._reset(reset)
        obs, reward = self._observe(reset)
        return arrdict.arrdict(
            obs=obs,
            reset=reset, 
            reward=reward)

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)

        self._lengths += 1

        reset = (self._lengths >= self._potential + 200)
        self._reset(reset)
        obs, reward = self._observe(reset)
        return arrdict.arrdict(
            obs=obs,
            reset=reset, 
            reward=reward) 

    def state(self, e=0):
        seen = self._seen[self._tex_to_env == e]
        return arrdict.arrdict(
            core=self.core.state(e),
            rgb=self._rgb.state(e),
            d=self._depth.state(e),
            potential=self._potential[e].clone(),
            seen=seen.clone(),
            length=self._lengths[e].clone(),
            max_length=self._potential[e].add(200).clone())

    @classmethod
    def plot_state(cls, state):
        fig = plt.figure()
        gs = plt.GridSpec(2, 2, fig, 0, 0, 1, 1)

        alpha = .1 + .9*state.seen.astype(float)
        # modifying this in place will bite me eventually. o for a lens
        state.core.scenery.textures.vals = np.concatenate([state.core.scenery.textures.vals, alpha[:, None]], 1)
        ax = core.Core.plot_state(state.core, plt.subplot(gs[:, 0]))

        images = {'rgb': state.rgb, 'd': state.d}
        plotting.plot_images(images, [plt.subplot(gs[:, 1])])

        s = (f'length: {state.length:d}/{state.max_length:.0f}\n'
            f'potential: {state.potential:.0f}')
        ax.annotate(s, (5., 5.), xycoords='axes points')

        return fig

    def display(self, d=0):
        return self.plot_state(arrdict.numpyify(self.state(d)))

