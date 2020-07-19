"""TODO-DOCS Demo docs"""
import torch
from megastep.demo import learning, lstm, heads
from rebar import logging, paths, stats, widgets, storing, arrdict, dotdict, recurrence, recording
import pandas as pd
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from megastep.demo.envs import explorer, deathmatch

log = logging.getLogger(__name__)

class Agent(nn.Module):

    def __init__(self, env, width=256):
        super().__init__()
        out = heads.output(env.action_space, width)
        self.sampler = out.sample
        self.policy = recurrence.Sequential(
            heads.intake(env.obs_space, width),
            lstm.LSTM(d_model=width),
            out)
        self.value = recurrence.Sequential(
            heads.intake(env.obs_space, width),
            lstm.LSTM(d_model=width),
            heads.ValueOutput(width))

    def forward(self, world, sample=False, value=False, test=False):
        outputs = arrdict.arrdict(
            logits=self.policy(world.obs, reset=world.reset))
        if sample or test:
            outputs['actions'] = self.sampler(outputs.logits, test)
        if value:
            outputs['value'] = self.value(world.obs, reset=world.reset)
        return outputs

def as_chunk(buffer):
    chunk = arrdict.stack(buffer)
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
    return chunk

def optimize(agent, opt, batch, entropy=1e-2, gamma=.99, clip=.2):
    w, d0 = batch.world, batch.decision
    d = agent(w, value=True)

    logits = learning.flatten(d.logits)
    old_logits = learning.flatten(learning.gather(d0.logits, d0.actions)).sum(-1)
    new_logits = learning.flatten(learning.gather(d.logits, d0.actions)).sum(-1)
    ratio = (new_logits - old_logits).exp().clamp(.05, 20)

    v_target = learning.v_trace(ratio, d.value, w.reward, w.reset, gamma=gamma)
    v_clipped = d0.value + torch.clamp(d.value - d0.value, -10, +10)
    v_loss = .5*torch.max((d.value - v_target)**2, (v_clipped - v_target)**2).mean()

    adv = learning.generalized_advantages(d.value, w.reward, d.value, w.reset, gamma=gamma)
    normed_adv = (adv - adv.mean())/(1e-3 + adv.std())
    free_adv = ratio*normed_adv
    clip_adv = torch.clamp(ratio, 1-clip, 1+clip)*normed_adv
    p_loss = -torch.min(free_adv, clip_adv).mean()

    h_loss = (logits.exp()*logits).sum(-1).mean()
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

def train():
    buffer_size = 32
    n_envs = 8*1024
    batch_size = 16*1024

    env = explorer.Explorer(n_envs)
    agent = Agent(env).cuda()
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4, amsgrad=True)

    run_name = f'{pd.Timestamp.now():%Y-%m-%d %H%M%S} {type(env).__name__}'
    paths.clear(run_name)
    compositor = widgets.Compositor()
    with logging.via_dir(run_name, compositor), stats.via_dir(run_name, compositor):
        
        world = env.reset()
        while True:
            buffer = []
            state = recurrence.get(agent)
            for _ in range(buffer_size):
                decision = agent(world[None], sample=True, value=True).squeeze(0).detach()
                buffer.append(arrdict.arrdict(
                    world=world,
                    decision=decision))
                buffer = buffer[-buffer_size:]
                world = env.step(decision)
                log.info('actor stepped')

            chunk = as_chunk(buffer)
            
            for idxs in learning.batch_indices(chunk, batch_size):
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
    env = explorer.Explorer(d+1) if env is None else env
    world = env.reset()
    if agent is None:
        agent = Agent(env).cuda()
        agent.load_state_dict(storing.load(run)['agent'], strict=False)

    world = env.reset()
    steps = 0
    with recording.ParallelEncoder(env.plot_state, N=N) as encoder, \
            tqdm(total=length) as pbar:
        while True:
            decision = agent(world[None], sample=True, test=test, value=True).squeeze(0)
            world = env.step(decision)
            steps += 1
            pbar.update(1)
            if length is None and world.reset.any():
                break
            state = env.state(d)
            encoder(arrdict.numpyify(arrdict.arrdict(**state, decision=decision)))
            if (steps == length):
                break
    encoder.notebook()
    return encoder