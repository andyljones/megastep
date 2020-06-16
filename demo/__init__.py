import torch
from torch import nn
from torch.nn import functional as F
from . import learning, agents
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict
import gym
import pandas as pd
import onedee
import cubicasa
import logging

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    ds = cubicasa.sample(n_envs)
    return onedee.ExplorationEnv(ds)

def agentfunc():
    env = envfunc(n_envs=1)
    return agents.Agent(env.observation_space, env.action_space).cuda()

def run():
    buffer_size = 100
    batch_size = 128

    env = envfunc(128)
    reaction = env.reset()
    agent = agentfunc().cuda()
    opt = torch.optim.Adam(agent.parameters(), lr=4.8e-4)

    buffer = []
    while True:
        decision = agent(reaction[None], sample=True).squeeze(0)
        buffer.append(arrdict(
            reaction=reaction,
            decision=decision))
        buffer = buffer[-buffer_size:]
        reaction = env.step(decision)

        if len(buffer) == buffer_size:
            chunk = arrdict.stack(buffer)
            batch = learning.sample(chunk, batch_size)
            learning.step(agent, opt, batch)
            
            display.clear_output(wait=True)
            print(chunk.reaction.reward.mean())


def demo():
    env = envfunc(1)
    reaction = env.reset()
    agent = agentfunc().cuda()

    reaction = env.reset()
    recorder = recording.recorder('test', env._plot, length=256, fps=20)
    while True:
        decisions = agent(reaction[None], sample=True).squeeze(0)
        reaction = env.step(decisions)

        recorder(env.state(0))
