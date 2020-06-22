import torch
from . import learning, agents
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict, dotdict
import pandas as pd
import onedee
from onedee import recording
import cubicasa
import numpy as np

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    return onedee.RandomChain(n_envs, n=10)
    # return onedee.ExplorerEnv(cubicasa.sample(n_envs))
    return onedee.WaypointEnv([cubicasa.column()]*n_envs)
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

def step(agent, opt, batch, entropy=1e-2, gamma=.99):
    decision = agent(batch.world, value=True)

    logits = learning.flatten(decision.logits)
    old_logits = learning.flatten(learning.gather(batch.decision.logits, batch.decision.actions)).sum(-1)
    new_logits = learning.flatten(learning.gather(decision.logits, batch.decision.actions)).sum(-1)
    ratios = (new_logits - old_logits).exp()

    reward = batch.world.reward
    rewardz = agent.scaler.scale(reward)
    valuez = decision.value
    value = agent.scaler.unnorm(valuez)
    reset = batch.world.reset
    terminal = batch.world.terminal

    # v = v_trace(ratios, value, reward, reset, terminal, gamma=gamma)
    v = learning.reward_to_go(reward, value, reset, terminal, gamma=gamma)
    vz = agent.scaler.norm(v)

    adv = learning.generalized_advantages(valuez, rewardz, vz, reset, terminal, gamma=gamma)

    v_loss = .5*(vz - valuez).pow(2).mean() 
    p_loss = (adv*new_logits[:-1]).mean()
    h_loss = -(logits.exp()*logits)[:-1].sum(-1).mean()
    loss = v_loss - p_loss - entropy*h_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.)

    opt.step()
    agent.scaler.step(v)

    with stats.defer():
        stats.mean('loss/value', v_loss)
        stats.mean('loss/policy', p_loss)
        stats.mean('loss/entropy', h_loss)
        stats.mean('loss/total', loss)
        stats.mean('resid-var/v', (v - value).pow(2).mean(), v.pow(2).mean())
        stats.mean('resid-var/vz', (vz - valuez).pow(2).mean(), vz.pow(2).mean())
        stats.mean('rel-entropy', -(logits.exp()*logits).sum(-1).mean()/np.log(logits.shape[-1]))
        stats.mean('debug-v/v', v.mean())
        stats.mean('debug-v/r-inf', reward.mean()/(1 - gamma))
        stats.mean('debug-scale/vz', vz.abs().mean())
        stats.mean('debug-scale/v', v.abs().mean())
        stats.mean('debug-max/v', v.abs().max())
        stats.mean('debug-scale/adv', adv.abs().mean())
        stats.mean('debug-max/adv', adv.abs().max())
        # stats.rel_gradient_norm('rel-norm-grad', agent)
        stats.mean('debug-scale/ratios', ratios.mean())
        stats.rate('rate/learner', reset.nelement())
        stats.rate('step-rate/learner', 1)
        stats.cumsum('steps/learner', 1)
        stats.last('scaler/mean', agent.scaler.mu)
        stats.last('scaler/std', agent.scaler.sigma)

def run():
    buffer_size = 128
    batch_size = 512
    n_envs = 512
    gearing = 8

    env = envfunc(n_envs)
    agent = agentfunc().cuda()
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)

    run_name = 'test'
    paths.clear(run_name)
    compositor = widgets.Compositor()
    with logging.via_dir(run_name, compositor), stats.via_dir(run_name, compositor):
        
        buffer = []
        world = env.reset()
        while True:
            for _ in range(gearing):
                decision = agent(world[None], sample=True).squeeze(0)
                buffer.append(arrdict(
                    world=world,
                    decision=decision))
                buffer = buffer[-buffer_size:]
                world = env.step(decision)
            
            if len(buffer) == buffer_size:
                chunk = arrdict.stack(buffer)
                chunkstats(chunk[-gearing:])

                batch = learning.sample(chunk, batch_size//buffer_size)
                step(agent, opt, batch)
                log.info('stepped')
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
