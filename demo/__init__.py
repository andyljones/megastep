import torch
from . import learning, agents
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict
import pandas as pd
import onedee
from onedee import recording
import cubicasa

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    return onedee.ObliviousEnv(n_envs)
    ds = cubicasa.sample(n_envs)
    return onedee.ExplorerEnv(ds)

def agentfunc():
    env = envfunc(n_envs=1)
    return agents.Agent(env.observation_space, env.action_space).cuda()

def chunkstats(chunk):
    stats.rate('rate/actor', chunk.world.reset.nelement())
    stats.mean('traj-length', chunk.world.reset.nelement(), chunk.world.reset.sum())
    stats.mean('step-reward', chunk.world.reward.sum(), chunk.world.reward.nelement())
    stats.mean('traj-reward', chunk.world.reward.sum(), chunk.world.reset.sum())

def run():
    buffer_size = 128
    batch_size = 2048
    n_envs = 2048
    gearing = 8

    env = envfunc(n_envs)
    agent = agentfunc().cuda()
    opt = torch.optim.Adam(agent.parameters(), lr=4.8e-3)

    paths.clear('test')
    compositor = widgets.Compositor()
    with logging.via_dir('test', compositor), stats.via_dir('test', compositor):
        
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
                learning.step(agent, opt, batch)
                log.info('stepped')
                stats.rate('rate/learner', batch_size)

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
