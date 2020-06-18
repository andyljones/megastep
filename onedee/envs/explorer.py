import torch
from .. import modules, core
from rebar import arrdict

class ExplorerEnv: 

    def __init__(self, *args, max_length=512, **kwargs):
        self._core = core.Core(*args, **kwargs)
        self._mover = modules.SimpleMovement(self._core)
        self._observer = modules.RGBDObserver(self._core)
        self._respawner = modules.RandomSpawns(self._core)

        self.action_space = self._mover.action_space
        self.observation_space = self._observer.observation_space

        self._tex_to_env = self._core.scene.lines.inverse[self._core.scene.textures.inverse.to(torch.long)].to(torch.long)
        self._seen = torch.full_like(self._tex_to_env, False)

        self._length = core.env_full_like(self._core, 0)
        self._max_length = torch.randint_like(self._length, max_length//2, max_length)

        self._potential = core.env_full_like(self._core, 0.)

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
        reward = reward.float()/self._core.res
        self._potential += reward
        return reward

    def _reset(self, reset):
        self._respawner(reset)
        self._seen[reset[self._tex_to_env]] = False
        self._length[reset] = 0
        self._potential[reset] = 0

    @torch.no_grad()
    def reset(self):
        reset = env_full_like(self._core, True)
        self._reset(reset)
        render = self._observer.render()
        return arrdict(
            obs=self._observer(render), 
            reset=reset, 
            terminal=reset, 
            reward=self._reward(render, reset))

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)
        self._length += 1

        reset = self._length == self._max_length
        self._reset(reset)
        render = self._observer.render()
        return arrdict(
            obs=self._observer(render), 
            reset=reset, 
            terminal=reset, 
            reward=self._reward(render, reset))

    def state(self, d=0):
        seen = self._seen[self._tex_to_env == d]
        return arrdict(
            **self._core.state(d),
            obs=self._observer.state(d),
            potential=self._potential[d].clone(),
            seen=seen.clone(),
            length=self._length[d].clone(),
            max_length=self._max_length[d].clone())

    @classmethod
    def plot_state(cls, state):
        fig = plt.figure()
        gs = plt.gridspec(2, 2, fig, 0, 0, 1, 1)

        alpha = .1 + .9*state.seen.astype(float)
        # modifying this in place will bite me eventually. o for a lens
        state['scene']['textures'] = np.concatenate([state.scene.textures, alpha[:, none]], 1)
        ax = plotting.plot_core(state, plt.subplot(gs[:, 0]))
        plotting.plot_images(state.obs, [plt.subplot(gs[0, 1])])

        s = (f'length: {state.length:d}/{state.max_length:d}\n'
            f'potential: {state.potential:.2f}')
        ax.annotate(s, (5., 5.), xycoords='axes points')

        return fig

    def display(self, d=0):
        return self.plot_state(arrdict.numpyify(self.state(d)))

