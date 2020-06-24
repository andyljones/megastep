import torch
from . import learning
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict, dotdict, recurrence
import pandas as pd
import onedee
from onedee import recording, spaces
import cubicasa
import numpy as np
import pandas as pd
from .transformer import Transformer
from torch import nn

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    # return onedee.RandomChain(n_envs, n=10)
    return onedee.ExplorerEnv(cubicasa.sample(n_envs))
    # return onedee.WaypointEnv([cubicasa.column()]*n_envs)
    ds = cubicasa.sample(n_envs)
    return onedee.ExplorerEnv(ds)

class Agent(nn.Module):

    def __init__(self, observation_space, action_space, width=128):
        super().__init__()
        out = spaces.output(action_space, width)
        self.sampler = out.sample
        self.policy = recurrence.Sequential(
            spaces.intake(observation_space, width),
            Transformer(mem_len=64, d_model=width, n_layers=2, n_head=2),
            out)
        self.value = recurrence.Sequential(
            spaces.intake(observation_space, width),
            Transformer(mem_len=64, d_model=width, n_layers=2, n_head=2),
            spaces.ValueOutput(width, 1))

        self.vnorm = learning.Normer()
        self.advnorm = learning.Normer()

    def forward(self, world, sample=False, value=False):
        outputs = arrdict(
            logits=self.policy(world.obs, reset=world.reset))
        if sample:
            outputs['actions'] = self.sampler(outputs.logits)
        if value:
            outputs['value_z'] = self.value(world.obs, reset=world.reset).squeeze(-1)
            outputs['value'] = self.vnorm.unnorm(outputs.value_z)
        return outputs

def agentfunc():
    env = envfunc(n_envs=1)
    return Agent(env.observation_space, env.action_space).cuda()

def chunkstats(chunk):
    with stats.defer():
        stats.rate('rate/actor', chunk.world.reset.nelement())
        stats.mean('traj-length', chunk.world.reset.nelement(), chunk.world.reset.sum())
        stats.cumsum('traj-count', chunk.world.reset.sum())
        stats.mean('step-reward', chunk.world.reward.sum(), chunk.world.reward.nelement())
        stats.mean('traj-reward', chunk.world.reward.sum(), chunk.world.reset.sum())

def optimize(agent, opt, batch, entropy=1e-2, gamma=.99, clip=.2):
    w, d0 = batch.world, batch.decision
    d = agent(w, value=True)

    logits = learning.flatten(d.logits)
    old_logits = learning.flatten(learning.gather(d0.logits, d0.actions)).sum(-1)
    new_logits = learning.flatten(learning.gather(d.logits, d0.actions)).sum(-1)
    ratio = (new_logits - old_logits).exp()

    rtg = learning.reward_to_go(w.reward, d0.value, w.reset, w.terminal, gamma=gamma)
    rtg_z = agent.vnorm.norm(rtg)
    v_clipped = d0.value_z + torch.clamp(d.value_z - d0.value_z, -clip, +clip)
    v_loss = .5*torch.max((d.value_z - rtg_z)**2, (v_clipped - rtg_z)**2).mean()

    adv = learning.generalized_advantages(d0.value, w.reward, d0.value, w.reset, w.terminal, gamma=gamma)
    adv_z = agent.advnorm.norm(adv)
    free_adv = ratio[:-1]*adv_z
    clip_adv = torch.clamp(ratio[:-1], 1-clip, 1+clip)*adv_z
    p_loss = -torch.min(free_adv, clip_adv).mean()

    h_loss = (logits.exp()*logits)[:-1].sum(-1).mean()
    loss = v_loss + p_loss + entropy*h_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.)

    opt.step()
    agent.vnorm.step(rtg)
    agent.advnorm.step(adv)

    kl_div = -(new_logits - old_logits).mean().detach()
    with stats.defer():
        stats.mean('loss/value', v_loss)
        stats.mean('loss/policy', p_loss)
        stats.mean('loss/entropy', h_loss)
        stats.mean('resid-var/v', (rtg - d.value).pow(2).mean(), rtg.pow(2).mean())
        stats.mean('rel-entropy', -(logits.exp()*logits).sum(-1).mean()/np.log(logits.shape[-1]))
        stats.mean('kl-div', kl_div) 
        stats.mean('debug-v/v', d.value.mean())
        stats.mean('debug-v/r-inf', w.reward.mean()/(1 - gamma))
        stats.mean('debug-scale/v', d.value.abs().mean())
        stats.mean('debug-max/v', d.value.abs().max())
        stats.mean('debug-scale/adv', adv.abs().mean())
        stats.mean('debug-max/adv', adv.abs().max())
        # stats.rel_gradient_norm('rel-norm-grad', agent)
        stats.rate('rate/learner', w.reset.nelement())
        stats.rate('step-rate/learner', 1)
        stats.cumsum('steps/learner', 1)

    return kl_div

def run():
    buffer_size = 64
    batch_size = 8192
    n_envs = 512

    env = envfunc(n_envs)
    agent = agentfunc().cuda()
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)

    run_name = f'{pd.Timestamp.now():%Y-%m-%d %H%M%S} test'
    paths.clear(run_name)
    compositor = widgets.Compositor()
    with logging.via_dir(run_name, compositor), stats.via_dir(run_name, compositor):
        
        buffer = []
        world = env.reset()
        while True:

            state = recurrence.get(agent)
            for _ in range(buffer_size):
                decision = agent(world[None], sample=True, value=True).squeeze(0).detach()
                buffer.append(arrdict(
                    world=world,
                    decision=decision))
                buffer = buffer[-buffer_size:]
                world = env.step(decision)
            
            chunk = arrdict.stack(buffer)
            chunkstats(chunk)

            indices = learning.batch_indices(n_envs, batch_size//buffer_size)
            for idxs in indices:
                with recurrence.temp_clear_set(agent, state[:, idxs]):
                    kl = optimize(agent, opt, chunk[:, idxs])

                log.info('stepped')
                if kl > .02:
                    log.info('kl div exceeded')
                    break
            learning.update_lr(opt)
            storing.store_latest(run_name, {'agent': agent}, throttle=60)
            stats.gpu.memory(0)
            stats.gpu.vitals(0)

def demo():
    env = envfunc(1)
    world = env.reset()
    agent = agentfunc().cuda()

    states = []
    world = env.reset()
    for _ in range(128):
        decision = agent(world[None], sample=True).squeeze(0)
        world = env.step(decision)
        states.append(env.state(0))
    states = arrdict.numpyify(states)
    recording.replay(env.plot_state, states)
