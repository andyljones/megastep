import time
import asyncio 
import torch
from rebar import queuing, processes, logging, stats, interrupting
from aljpy import arrdict, recording
import numpy as np
import aljpy

log = aljpy.logger()

async def chunk(envfunc, agentfunc, run_name, queues, canceller, chunklength=20):
    async with logging.to_dir(run_name), \
            interrupting.interrupter(), \
            stats.to_dir(run_name), \
            queuing.cleanup(queues.agents, queues.chunks):

        env, agent = envfunc(), agentfunc()

        prev_reaction = env.reset()
        while True:
            chunk = []
            for _ in range(chunklength):
                state_dict = queues.agents.get()
                if state_dict is not None:
                    agent.load_state_dict(state_dict)
                    log.info('Updated agent')
                stats.timeaverage('queues/agents-empty', state_dict is None)

                decision = agent(prev_reaction.unsqueeze(0), sample=True).squeeze(0)
                next_reaction = env.step(decision)
                chunk.append(arrdict.arrdict(
                    prev_reaction=prev_reaction,
                    decision=decision).detach())
                prev_reaction = next_reaction

                await processes.surrender()

            chunk = arrdict.stack(chunk, 0)
            success = queues.chunks.put(chunk)
            if success:
                log.info('Chunk sent')
            else:
                log.info('Chunk dropped')
            stats.timeaverage('queues/chunks-full', not success)

            time.sleep(.001)

            r = chunk.prev_reaction
            stats.mean('debug-scale/obs', r.obs.abs().float().mean())
            stats.max('debug-max/reward', r.reward.abs().max())
            stats.mean('reward-per-sample', r.reward.mean())
            stats.mean('reward-per-traj', r.reward.sum(), r.terminal.sum())
            stats.cumsum('samples/actor', r.reward.nelement())
            stats.rate('sample-rate/actor', r.reward.nelement())
            stats.rate('step-rate/actor', 1)
            stats.cumsum('steps/actor', 1)
            stats.cumsum('steps/env', r.obs.size(0))
            stats.cumsum('steps/traj', r.terminal.sum())
            stats.mean('traj-length', r.terminal.nelement(), r.terminal.sum())

            if canceller.is_set():
                break
    
def record(env, agent, steps, fps=20):
    prev_reaction = env.reset()
    with recording.Encoder(fps) as encoder:
        for s in range(steps):
            encoder(env.render())
            decision = agent(prev_reaction.unsqueeze(0), sample=True).squeeze(0)
            prev_reaction = env.step(decision)
    return recording.notebook(encoder)

def rollout(env, agent, steps):
    prev_reaction = env.reset()
    results = []
    for s in range(steps):
        decision = agent(prev_reaction.unsqueeze(0), sample=True, value=True).squeeze(0)
        next_reaction = env.step(decision)
        results.append(arrdict.arrdict(
            prev_reaction=prev_reaction,
            decision=decision,
            next_reaction=next_reaction))
        prev_reaction = next_reaction
    return arrdict.stack(results)
    