import torch
from torch import nn
from torch.nn import functional as F
from . import acting, learning, intakes
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing
import aljpy
import gym
from aljpy import arrdict
import pandas as pd

log = aljpy.logger()

def categorical_sample(l):
    samples = torch.distributions.Categorical(logits=l.float().reshape(-1, l.shape[-1])).sample()
    return samples.reshape(l.shape[:-1])

class Agent(nn.Module):

    def __init__(self, observation_space, action_space, width=128):
        super().__init__()
        self.intake = intakes.intake(observation_space, width)
        self.torso = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU())
        self.policy = nn.Linear(width, action_space.n)
        self.value = nn.Linear(width, 1)

        self.register_buffer('gen', torch.tensor(0))

    def forward(self, reaction, sample=False, value=False):
        x = self.intake(reaction.obs)
        x = self.torso(x)

        outputs = arrdict.arrdict(
            gen=self.gen.new_full(x.shape[:2], self.gen),
            log_likelihood=F.log_softmax(self.policy(x), dim=-1))
        if sample:
            outputs['action'] = categorical_sample(outputs.log_likelihood)
        if value:
            outputs['value'] = self.value(x).squeeze(-1)
        return outputs

def envfunc(n_envs=1024):
    # return CartPoleEnv(n_envs)
    # return statemachines.encode(statemachines.double_random_chain(4), n_envs)
    return atari.ImageEnv('PongNoFrameskip-v4', n_envs, frameskip=4)

def agentfunc():
    env = envfunc(n_envs=1)
    return Agent(env.observation_space, env.action_space).cuda()

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
