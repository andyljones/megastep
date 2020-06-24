import torch
from . import learning, agents
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict, dotdict, recurrence
import pandas as pd
import onedee
from onedee import recording
import cubicasa
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    # return onedee.RandomChain(n_envs, n=10)
    return onedee.ExplorerEnv(cubicasa.sample(n_envs))
    # return onedee.WaypointEnv([cubicasa.column()]*n_envs)
    ds = cubicasa.sample(n_envs)
    return onedee.ExplorerEnv(ds)

def agentfunc():
    env = envfunc(n_envs=1)
    return agents.Agent(env.observation_space, env.action_space).cuda()

def chunkstats(chunk):
    with stats.defer():
        stats.rate('rate/actor', chunk.world.reset.nelement())
        stats.mean('traj-length', chunk.world.reset.nelement(), chunk.world.reset.sum())
        stats.cumsum('traj-count', chunk.world.reset.sum())
        stats.mean('step-reward', chunk.world.reward.sum(), chunk.world.reward.nelement())
        stats.mean('traj-reward', chunk.world.reward.sum(), chunk.world.reset.sum())

def step(agent, opt, batch, entropy=1e-2, gamma=.99, clip=.2):
    decision = agent(batch.world, value=True)

    logits = learning.flatten(decision.logits)
    old_logits = learning.flatten(learning.gather(batch.decision.logits, batch.decision.actions)).sum(-1)
    new_logits = learning.flatten(learning.gather(decision.logits, batch.decision.actions)).sum(-1)
    ratio = (new_logits - old_logits).exp()

    reward = batch.world.reward
    reset = batch.world.reset
    terminal = batch.world.terminal
    value0 = batch.decision.value
    value = decision.value

    rtg = learning.reward_to_go(reward, value, reset, terminal, gamma=gamma)
    v_clipped = value0 + torch.clamp(value - value0, -clip, +clip)
    v_loss = .5*torch.max((value - rtg)**2, (v_clipped - rtg)**2).mean()

    adv = learning.generalized_advantages(value0, reward, value0, reset, terminal, gamma=gamma)
    free_adv = ratio[:-1]*adv
    clip_adv = torch.clamp(ratio[:-1], 1-clip, 1+clip)*adv
    p_loss = -torch.min(free_adv, clip_adv).mean()

    h_loss = (logits.exp()*logits)[:-1].sum(-1).mean()
    loss = v_loss + p_loss + entropy*h_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.)

    opt.step()

    kl_div = -(new_logits - old_logits).mean().detach()
    with stats.defer():
        stats.mean('loss/value', v_loss)
        stats.mean('loss/policy', p_loss)
        stats.mean('loss/entropy', h_loss)
        stats.mean('loss/total', loss)
        stats.mean('resid-var/v', (rtg - value).pow(2).mean(), rtg.pow(2).mean())
        stats.mean('rel-entropy', -(logits.exp()*logits).sum(-1).mean()/np.log(logits.shape[-1]))
        stats.mean('kl-div', kl_div) 
        stats.mean('debug-v/v', value.mean())
        stats.mean('debug-v/r-inf', reward.mean()/(1 - gamma))
        stats.mean('debug-scale/v', value.abs().mean())
        stats.mean('debug-max/v', value.abs().max())
        stats.mean('debug-scale/adv', adv.abs().mean())
        stats.mean('debug-max/adv', adv.abs().max())
        # stats.rel_gradient_norm('rel-norm-grad', agent)
        stats.rate('rate/learner', reset.nelement())
        stats.rate('step-rate/learner', 1)
        stats.cumsum('steps/learner', 1)

    return kl_div

def run():
    buffer_size = 32
    batch_size = 4096
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
            
            if len(buffer) == buffer_size:
                chunk = arrdict.stack(buffer)
                chunkstats(chunk)

                indices = learning.batch_indices(n_envs, batch_size//buffer_size)
                for idxs in indices:
                    with recurrence.temp_clear_set(agent, state[:, idxs]):
                        kl = step(agent, opt, chunk[:, idxs])

                    log.info('stepped')
                    if kl > .02:
                        log.info('kl div exceeded')
                        break
                storing.store(run_name, {'agent': agent}, throttle=60)

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
