from onedee.plotting import n_agent_texels
import torch
from . import learning, lstm, transformer
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict, dotdict, recurrence, recording
import pandas as pd
import onedee
from onedee import spaces
import cubicasa
import numpy as np
import pandas as pd
from torch import nn
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    # return onedee.DelayedMatchCoin(n_envs)
    return onedee.Deathmatch(cubicasa.sample(max(n_envs, 4)//4), n_agents=4)
    # return onedee.Waypoint(cubicasa.sample(n_envs))
    return onedee.PointGoal(cubicasa.sample(n_envs))
    # return onedee.Explorer(cubicasa.sample(n_envs))

def init_lstm(lstm, forget=1.):
    nn.init.orthogonal_(2**-.5 * lstm.weight_ih_l0)
    nn.init.orthogonal_(2**-.5 * lstm.weight_hh_l0)

    # Order is input, forget, cell, output
    for bias in [lstm.bias_ih_l0, lstm.bias_hh_l0]:
        i, f, c, o = bias.chunk(4, 0)
        nn.init.constant_(i, 0)
        nn.init.constant_(f, forget/2.)
        nn.init.constant_(c, 0)
        nn.init.constant_(o, 0)

class Agent(nn.Module):

    def __init__(self, observation_space, action_space, width=256):
        super().__init__()
        out = spaces.output(action_space, width)
        self.sampler = out.sample
        self.policy = recurrence.Sequential(
            spaces.intake(observation_space, width),
            lstm.LSTM(d_model=width),
            # transformer.Transformer(mem_len=128, d_model=width),
            out)
        self.value = recurrence.Sequential(
            spaces.intake(observation_space, width),
            lstm.LSTM(d_model=width),
            # transformer.Transformer(mem_len=128, d_model=width),
            spaces.ValueOutput(width, 1))

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.LSTM):
            init_lstm(m)

    def forward(self, world, sample=False, value=False, test=False):
        outputs = arrdict(
            logits=self.policy(world.obs, reset=world.reset))
        if sample or test:
            outputs['actions'] = self.sampler(outputs.logits, test)
        if value:
            outputs['value'] = self.value(world.obs, reset=world.reset).squeeze(-1)
        return outputs

class MultiAgent(nn.Module):

    def __init__(self, *args, n_agents=4, **kwargs):
        super().__init__()
        self.agents = nn.ModuleList([Agent(*args, **kwargs) for i in range(n_agents)])

    def forward(self, world, **kwargs):
        return arrdict.stack([agent(world[:, [i]], **kwargs) for i, agent in enumerate(self.agents)])

def agentfunc():
    env = envfunc(n_envs=1)
    return Agent(env.observation_space, env.action_space).cuda()

def chunkstats(chunk):
    with stats.defer():
        stats.rate('sample-rate/actor', chunk.world.reset.nelement())
        stats.mean('traj-length', chunk.world.reset.nelement(), chunk.world.reset.sum())
        stats.cumsum('count/traj', chunk.world.reset.sum())
        stats.cumsum('count/world', chunk.world.reset.size(0))
        stats.cumsum('count/chunks', 1)
        stats.cumsum('count/samples', chunk.world.reset.nelement())
        stats.rate('step-rate/chunks', 1)
        stats.rate('step-rate/world', chunk.world.reset.size(0))
        stats.mean('step-reward', chunk.world.reward.sum(), chunk.world.reward.nelement())
        stats.mean('traj-reward/mean', chunk.world.reward.sum(), chunk.world.reset.sum())
        stats.mean('traj-reward/positive', chunk.world.reward.clamp(0, None).sum(), chunk.world.reset.sum())
        stats.mean('traj-reward/negative', chunk.world.reward.clamp(None, 0).sum(), chunk.world.reset.sum())

def optimize(agent, opt, batch, entropy=1e-3, gamma=.995, clip=.2):
    w, d0 = batch.world, batch.decision
    d = agent(w, value=True)

    logits = learning.flatten(d.logits)
    old_logits = learning.flatten(learning.gather(d0.logits, d0.actions)).sum(-1)
    new_logits = learning.flatten(learning.gather(d.logits, d0.actions)).sum(-1)
    ratio = (new_logits - old_logits).exp().clamp(.05, 20)

    v_target = learning.v_trace(ratio, d.value, w.reward, w.reset, w.terminal, gamma=gamma)
    v_clipped = d0.value + torch.clamp(d.value - d0.value, -10, +10)
    v_loss = .5*torch.max((d.value - v_target)**2, (v_clipped - v_target)**2).mean()

    adv = learning.generalized_advantages(d0.value, w.reward, d0.value, w.reset, w.terminal, gamma=gamma)
    normed_adv = (adv - adv.mean())/(1e-3 + adv.std())
    free_adv = ratio[:-1]*normed_adv
    clip_adv = torch.clamp(ratio[:-1], 1-clip, 1+clip)*normed_adv
    p_loss = -torch.min(free_adv, clip_adv).mean()

    h_loss = (logits.exp()*logits)[:-1].sum(-1).mean()
    loss = v_loss + p_loss + entropy*h_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 100.)
    torch.nn.utils.clip_grad_norm_(agent.value.parameters(), 100.)

    opt.step()

    kl_div = -(new_logits - old_logits).mean().detach()
    with stats.defer():
        stats.mean('loss/value', v_loss)
        stats.mean('loss/policy', p_loss)
        stats.mean('loss/entropy', h_loss)
        stats.mean('resid-var/v', (v_target - d.value).pow(2).mean(), v_target.pow(2).mean())
        stats.mean('rel-entropy', -(logits.exp()*logits).sum(-1).mean()/np.log(logits.shape[-1]))
        stats.mean('kl-div', kl_div) 

        stats.mean('v-target/mean', v_target.mean())
        stats.mean('v-target/std', v_target.std())

        stats.mean('adv/z-mean', adv.mean())
        stats.mean('adv/z-std', adv.std())
        stats.max('adv/z-max', adv.abs().max())

        stats.rate('sample-rate/learner', w.reset.nelement())
        stats.rate('step-rate/learner', 1)
        stats.cumsum('count/learner-steps', 1)
        # stats.rel_gradient_norm('rel-norm-grad', agent)

        stats.mean('param/gamma', gamma)
        stats.mean('param/entropy', entropy)

    return kl_div

def run():
    buffer_size = 32
    n_envs = 4096
    batch_size = 1*n_envs

    env = envfunc(n_envs)
    agent = agentfunc().cuda()
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4, amsgrad=True)

    run_name = f'{pd.Timestamp.now():%Y-%m-%d %H%M%S} test'
    paths.clear(run_name)
    compositor = widgets.Compositor()
    with logging.via_dir(run_name, compositor), stats.via_dir(run_name, compositor):
        
        steps = 0
        indices = learning.batch_indices(n_envs, batch_size//buffer_size)
        world = env.reset()
        while True:
            buffer = []
            state = recurrence.get(agent)
            for _ in range(buffer_size):
                decision = agent(world[None], sample=True, value=True).squeeze(0).detach()
                buffer.append(arrdict(
                    world=world,
                    decision=decision))
                buffer = buffer[-buffer_size:]
                world = env.step(decision)
                log.info('actor stepped')

            chunk = arrdict.stack(buffer)
            chunkstats(chunk)
            
            for _ in range((buffer_size*n_envs)//batch_size):
                idxs = indices[steps % len(indices)]
                steps += 1
                with recurrence.temp_clear_set(agent, state[:, idxs]):
                    kl = optimize(agent, opt, chunk[:, idxs])

                log.info('learner stepped')
                if kl > .02:
                    log.info('kl div exceeded')
                    break
            storing.store_latest(run_name, {'agent': agent}, throttle=60)
            stats.gpu.memory(0)
            stats.gpu.vitals(0, throttle=15)

def demo(run=-1, length=None, test=True, N=None, env=None, agent=None, d=0):
    env = envfunc(d+1) if env is None else env
    world = env.reset()
    if agent is None:
        agent = agentfunc().cuda()
        agent.load_state_dict(storing.load()['agent'], strict=False)

    world = env.reset()
    steps = 0
    traces = []
    with recording.ParallelEncoder(env.plot_state, N=N) as encoder, \
            tqdm(total=length) as pbar:
        while True:
            decision = agent(world[None], sample=True, test=test).squeeze(0)
            world = env.step(decision)
            steps += 1
            pbar.update(1)
            if length is None and world.reset.any():
                break
            if (steps == length):
                break
            encoder(arrdict.numpyify(env.state()))
            traces.append(arrdict.numpyify(arrdict(
                state=env.state(d), 
                world=world[d], 
                decision=decision[d])))
    traces = arrdict.stack(traces)
    encoder.notebook()
