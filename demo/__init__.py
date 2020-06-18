import torch
from . import learning, agents
from rebar import queuing, processes, logging, interrupting, paths, stats, widgets, storing, arrdict
import pandas as pd
import onedee
from onedee import recording
import cubicasa

log = logging.getLogger(__name__)

def envfunc(n_envs=1024):
    return onedee.WaypointEnv([cubicasa.column()]*n_envs)
    ds = cubicasa.sample(n_envs)
    return onedee.ExplorerEnv(ds)

def agentfunc():
    env = envfunc(n_envs=1)
    return agents.Agent(env.observation_space, env.action_space).cuda()

def chunkstats(chunk):
    stats.rate('rate/actor', chunk.reaction.reset.nelement())
    stats.mean('traj-length', chunk.reaction.reset.nelement(), chunk.reaction.reset.sum())
    stats.mean('step-reward', chunk.reaction.reward.sum(), chunk.reaction.reward.nelement())
    stats.mean('traj-reward', chunk.reaction.reward.sum(), chunk.reaction.reset.sum())

def run():
    buffer_size = 64
    batch_size = 4096
    n_envs = 4096
    gearing = 1

    env = envfunc(n_envs)
    agent = agentfunc().cuda()
    opt = torch.optim.Adam(agent.parameters(), lr=4.8e-3)

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
            
            if len(buffer) == buffer_size:
                chunk = arrdict.stack(buffer)
                chunkstats(chunk[-gearing:])

                batch = learning.sample(chunk, batch_size//buffer_size)
                learning.step(agent, opt, batch)
                log.info('stepped')
                stats.rate('rate/learner', batch_size)



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
    return recording.replay(env.plot_state, states)
