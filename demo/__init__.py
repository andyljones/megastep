import torch
from torch import nn
from torch.nn import functional as F
from . import acting, learning, agents
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict
import gym
import pandas as pd
from onedee import Environment
import designs
import logging

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    ds = designs.cubicasa(n_envs)
    return Environment(ds)

def agentfunc():
    env = envfunc(n_envs=1)
    return agents.Agent(env.observation_space, env.action_space).cuda()

def run():
    processes.set_start_method()
    torch.set_num_threads(1)

    run = f'{pd.Timestamp.now():%Y-%m-%d %H-%M-%S} impala-pong'
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
    agent = agentfunc()
    env = envfunc(n_envs=1)

    agent.load_state_dict(storing.load_one(procname='learn-0')['agent'])

    acting.record(env, agent, 256)
