import torch
from torch import nn
from torch.nn import functional as F
from . import acting, learning, agents
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict
import gym
import pandas as pd
from onedee import MinimalEnv, recording
import cubicasa
import logging

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    ds = cubicasa.sample(n_envs)
    return MinimalEnv(ds)

def agentfunc():
    env = envfunc(n_envs=1)
    return agents.Agent(env.observation_space, env.action_space).cuda()

def run():
    processes.set_start_method()
    torch.set_num_threads(1)

    run = f'{pd.Timestamp.now():%Y-%m-%d %H-%M-%S} demo'
    compositor = widgets.Compositor()
    with logging.from_dir(run, compositor), \
            stats.from_dir(run, compositor), \
            interrupting.interrupter() as interrupter, \
            processes.sentinel(serial=True) as sentinel:

        queues = queuing.create(('chunks', 'agents'), serial=sentinel.serial)

        sentinel.launch(acting.chunk, envfunc, agentfunc,
                run, queues, sentinel.canceller)
        sentinel.launch(learning.learn, agentfunc,
                run, queues, sentinel.canceller)

        while True:
            sentinel.check()
            interrupter.check()

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
