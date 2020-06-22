import torch
from . import learning, agents
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict, dotdict
import pandas as pd
import onedee
from onedee import recording
import cubicasa

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    return onedee.RandomChain(n_envs)
    return onedee.WaypointEnv([cubicasa.column()]*n_envs)
    ds = cubicasa.sample(n_envs)
    return onedee.ExplorerEnv(ds)

def agentfunc():
    env = envfunc(n_envs=1)
    return agents.Agent(env.observation_space, env.action_space, width=16).cuda()

def chunkstats(chunk):
    with stats.defer():
        stats.rate('rate/actor', chunk.world.reset.nelement())
        stats.mean('traj-length', chunk.world.reset.nelement(), chunk.world.reset.sum())
        stats.cumsum('n-traj', chunk.world.reset.sum())
        stats.mean('step-reward', chunk.world.reward.sum(), chunk.world.reward.nelement())
        stats.mean('traj-reward', chunk.world.reward.sum(), chunk.world.reset.sum())

def stepstats(l):
    with stats.defer():
        stats.mean('loss/value', l.v_loss)
        stats.mean('loss/policy', l.p_loss)
        stats.mean('loss/entropy', l.h_loss)
        stats.mean('loss/total', l.loss)
        stats.mean('resid-var/v', (l.v - l.value).pow(2).mean(), l.v.pow(2).mean())
        stats.mean('resid-var/vz', (l.vz - l.valuez).pow(2).mean(), l.vz.pow(2).mean())
        stats.mean('entropy', -(l.logits.exp()*l.logits).sum(-1).mean())
        stats.mean('debug-v/v', l.v.mean())
        stats.mean('debug-v/r-inf', l.reward.mean()/(1 - l.gamma))
        stats.mean('debug-scale/vz', l.vz.abs().mean())
        stats.mean('debug-scale/v', l.v.abs().mean())
        stats.mean('debug-max/v', l.v.abs().max())
        stats.mean('debug-scale/adv', l.adv.abs().mean())
        stats.mean('debug-max/adv', l.adv.abs().max())
        # stats.rel_gradient_norm('rel-norm-grad', l.agent)
        stats.mean('debug-scale/ratios', l.ratios.mean())
        stats.rate('rate/learner', l.reset.nelement())
        stats.rate('step-rate/learner', 1)
        stats.cumsum('steps/learner', 1)
        stats.last('scaler/mean', l.agent.scaler.mu)
        stats.last('scaler/std', l.agent.scaler.sigma)



def step(agent, opt, batch, entropy=.01, gamma=.99):
    decision = agent(batch.world, value=True)

    logits = learning.flatten(decision.logits)
    new_logits = learning.flatten(learning.gather(decision.logits, batch.decision.actions)).sum(-1)
    old_logits = learning.flatten(learning.gather(batch.decision.logits, batch.decision.actions)).sum(-1)
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

    v_loss = .5*(vz - valuez).pow(2).sum() 
    p_loss = (adv*new_logits[:-1]).sum()
    h_loss = -(logits.exp()*logits)[:-1].sum(-1).sum()
    loss = v_loss - p_loss - entropy*h_loss
    
    opt.zero_grad()
    loss.backward()

    agent.scaler.step(v)
    opt.step()
    # stepstats(dotdict(locals()))

def run():
    buffer_size = 16
    batch_size = 512
    n_envs = 512
    gearing = 8

    env = envfunc(n_envs)
    agent = agentfunc().cuda()
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)

    paths.clear('test')
    compositor = widgets.Compositor()
    with logging.via_dir('test', compositor), stats.via_dir('test', compositor):
        
        buffer = []
        world = env.reset()
        for _ in range(180):
            for _ in range(gearing):
                decision = agent(world[None], sample=True).squeeze(0)
                buffer.append(arrdict(
                    world=world,
                    decision=decision))
                buffer = buffer[-buffer_size:]
                world = env.step(decision)
            
            if len(buffer) == buffer_size:
                chunk = arrdict.stack(buffer)
                # chunkstats(chunk[-gearing:])

                batch = learning.sample(chunk, batch_size//buffer_size)
                step(agent, opt, batch)
                log.info('stepped')


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
