import torch
from torch import nn
from torch.nn import functional as F
from . import learning, agents
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict
import gym
import pandas as pd
import onedee
from onedee import recording
import cubicasa

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    ds = cubicasa.sample(n_envs)
    return onedee.ExplorationEnv(ds)

def agentfunc():
    env = envfunc(n_envs=1)
    return agents.Agent(env.observation_space, env.action_space).cuda()

def run():
    buffer_size = 64
    batch_size = 64
    n_envs = 4096
    gearing = 1

    env = envfunc(n_envs)
    agent = agentfunc().cuda()
    opt = torch.optim.Adam(agent.parameters(), lr=4.8e-4)

    paths.clear('test')
    compositor = widgets.Compositor()
    with logging.via_dir('test', compositor), stats.via_dir('test', compositor):
        
        buffer = []
        reaction = env.reset()
        while True:
            for _ in range(gearing):
                decision = agent(reaction[None], sample=True).squeeze(0)
                buffer.append(arrdict(
                    reaction=reaction,
                    decision=decision))
                buffer = buffer[-buffer_size:]
                reaction = env.step(decision)
                stats.rate('rate/actor', n_envs)
                stats.mean('traj-length', n_envs, reaction.reset.sum())
                stats.mean('step-reward', reaction.reward.sum(), n_envs)
                stats.mean('traj-reward', reaction.reward.sum(), reaction.reset.sum())
                
            
            if len(buffer) == buffer_size:
                chunk = arrdict.stack(buffer)
                batch = learning.sample(chunk, batch_size)
                learning.step(agent, opt, batch)
                log.info('stepped')
                stats.rate('rate/learner', batch_size*buffer_size)


def demo():
    env = envfunc(1)
    reaction = env.reset()
    agent = agentfunc().cuda()

    states = []
    reaction = env.reset()
    for _ in range(128):
        decisions = agent(reaction[None], sample=True).squeeze(0)
        reaction = env.step(decisions)
        states.append(env.state(0))
    states = arrdict.numpyify(arrdict.stack(states))
    return recording.replay(env._plot, states)
