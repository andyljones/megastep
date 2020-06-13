""" 
IMPALA structure:
    * You get a chunk of experience from the actor
    * You want to calculate gradients for both the value function and the policy 
    * The value function gradient depends on the current target value of 
"""
import numpy as np
import torch
import aljpy
from rebar import logging, queuing, processes, stats, storing, interrupting
import time

log = aljpy.logger()

def gather(arr, indices):
    if isinstance(arr, dict):
        return arr.__class__({k: gather(arr[k], indices[k]) for k in arr})
    return torch.gather(arr, -1, indices.type(torch.long).unsqueeze(-1)).squeeze(-1)

def deltas(reward, value, reset, terminal, gamma=.99):
    # Value comes from the decision before the reaction
    # Reward, reset and terminal come from the reaction to the decision
    #
    # If ~terminal[t] & ~reset[t]:
    #   * delta = reward[t] - (value[t] - gamma*value[t+1])
    #   * Because it's the error between the reward you actually got and what's inferred from the value
    # If terminal[t]: 
    #   * delta = reward - value[t]
    #   * Because value[t+1] is from another trajectory
    # If reset[t] & ~terminal[t]: 
    #   * delta = 0
    #   * Because we don't know what the successor value is, and the most innocent option is to
    #     assume it accounts for the reward perfectly 
    #
    # Because we're missing the value for the state after the final reward, we drop that final reward
    #
    # Indices of the output correspond to the front T-1 indices of the values. 
    reward, reset, terminal = reward[:-1], reset[:-1], terminal[:-1]
    regular_deltas = reward - (value[:-1] - gamma*value[1:])
    terminated_deltas = torch.where(terminal, reward - value[:-1], regular_deltas)
    return torch.where(reset & ~terminal, torch.zeros_like(reward), terminated_deltas)

def v_trace(ratios, value, reward, terminal, gamma, max_rho=1, max_c=1):
    rho = ratios.clamp(0, max_rho)
    c = ratios.clamp(0, max_c)
    dV = rho[:-1]*deltas(reward, value, terminal, terminal, gamma=gamma)

    discount = (1 - terminal.int())[:-1]*gamma

    v = torch.zeros_like(value)
    v[-1] = value[-1]
    for t in reversed(range(len(v)-1)):
        v[t] = value[t] + dV[t] + discount[t]*c[t]*(v[t+1] - value[t+1])

    return v.detach()

def advantages(ratios, value, reward, terminal, v, gamma, max_pg_rho=1):
    rho = ratios.clamp(0, max_pg_rho)
    discount = (1 - terminal.int())*gamma
    vprime = torch.cat([v[1:], value[[-1]]])
    adv = reward + discount*vprime - value
    return (rho*adv).detach()

def flat_params(agent):
    return torch.cat([p.data.float().flatten() for p in agent.parameters()]) 

def step(agent, opt, chunk, entropy=.01, gamma=.99):
    decision = agent(chunk.prev_reaction, value=True)

    log_ratios = decision.log_likelihood - chunk.decision.log_likelihood
    log_ratios = gather(log_ratios, chunk.decision.action)
    ratios = log_ratios.exp()

    reward, terminal = chunk.next_reaction.reward, chunk.next_reaction.terminal
    reward = reward.clamp(-1, +1)
    v = v_trace(ratios, decision.value, reward, terminal, gamma=gamma)
    adv = advantages(ratios, decision.value, reward, terminal, v, gamma=gamma)

    v_loss = .5*(v - decision.value).pow(2).mean() 
    p_loss = (adv*gather(decision.log_likelihood, chunk.decision.action)).mean()
    h_loss = -(decision.log_likelihood.exp()*decision.log_likelihood).sum(-1).mean()
    loss = v_loss - p_loss - entropy*h_loss

    original = flat_params(agent)
    
    opt.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(agent.parameters(), 40.)

    opt.step()

    stats.mean('loss/value', v_loss)
    stats.mean('loss/policy', p_loss)
    stats.mean('loss/entropy', h_loss)
    stats.mean('loss/total', loss)
    stats.mean('resid-var', (v - decision.value).pow(2).mean(), v.pow(2).mean())
    stats.mean('rel-entropy', -(decision.log_likelihood.exp()*decision.log_likelihood).mean()/np.log(decision.log_likelihood.size(-1)))
    stats.mean('debug-v/v', v.mean())
    stats.mean('debug-v/r-inf', reward.mean()/(1 - gamma))
    stats.mean('debug-scale/v', v.abs().mean())
    stats.mean('debug-max/v', v.abs().max())
    stats.mean('debug-scale/adv', adv.abs().mean())
    stats.mean('debug-max/adv', adv.abs().max())
    stats.rel_gradient_norm('rel-norm-grad', agent)
    stats.mean('gen-lag', agent.gen - chunk.decision.gen.float().mean())
    stats.mean('debug-scale/ratios', ratios.mean())
    stats.rate('step-rate/learner', 1)
    stats.cumsum('steps/learner', 1)

    new = flat_params(agent)
    stats.mean('rel-norm-update', (new - original).pow(2).mean(), original.pow(2).mean())

    agent.gen += 1

def sample(chunk, batchsize):
    B = chunk.prev_reaction.obs.shape[1]
    indices = torch.randint(B, size=(batchsize,), device=chunk.prev_reaction.obs.device)
    batch = chunk[:, indices]
    batch = type(batch)({**batch[:-1], 'next_reaction': batch[1:].prev_reaction})
    return batch

async def learn(agentfunc, run_name, queues, canceller):
    async with logging.to_dir(run_name), \
            interrupting.interrupter(), \
            stats.to_dir(run_name), \
            queuing.cleanup(queues.chunks, queues.agents):

        agent = agentfunc()
        opt = torch.optim.Adam(agent.parameters(), lr=4.8e-4)

        chunk = None
        while True:
            chunk = queues.chunks.get()
            stats.timeaverage('queues/chunks-empty', chunk is None)

            if chunk is not None:
                for _ in range(chunk.prev_reaction.obs.size(1)//128):
                    batch = sample(chunk, 128)
                    step(agent, opt, batch)
                    log.info('Stepped optimizer')
                    success = queues.agents.put({k: v.clone() for k, v in agent.state_dict().items()})
                    stats.timeaverage('queues/agents-full', success)

                    stats.cumsum('samples/learner', batch.next_reaction.reward.nelement())
                    stats.rate('sample-rate/learner', batch.next_reaction.reward.nelement())

                    time.sleep(.001)
                    await processes.surrender()

            await processes.surrender()

            storing.store(run_name, {'agent': agent}, throttle=60)

            if canceller.is_set():
                break


### TESTS ###

def explicit_v_trace(ratios, value, reward, terminal, gamma=.99, max_rho=1, max_c=1):
    rho = ratios.clamp(0, max_rho)
    c = ratios.clamp(0, max_c)

    v = value.clone()
    for s in range(len(v)-1):
        for t in range(s, len(v)-1):
            prod_c = c[s:t].prod()
            if not terminal[t]:
                dV = rho[t]*(reward[t] + gamma*value[t+1] - value[t])
                v[s] += gamma**(t - s) * prod_c*dV
            else:
                dV = rho[t]*(reward[t] - value[t])
                v[s] += gamma**(t - s) * prod_c*dV
                break
    
    return v

def test_v_trace_trivial():
    ratios = torch.tensor([1., 1.])
    value = torch.tensor([2., 3.])
    reward = torch.tensor([1., 1.])
    terminal = torch.tensor([False, False])
    gamma = 1.

    expected = explicit_v_trace(ratios, value, reward, terminal, gamma)
    torch.testing.assert_allclose(expected, torch.tensor([4., 3.]))

    actual = v_trace(ratios, value, reward, terminal, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([4., 3.]))

def test_v_trace_random(R=100, T=10):
    for _ in range(R):
        ratios = torch.rand((T,))
        value = torch.rand((T,))
        reward = torch.rand((T,))
        terminal = torch.rand((T,)) > .8
        gamma = torch.rand(())

        expected = explicit_v_trace(ratios, value, reward, terminal, gamma)
        actual = v_trace(ratios, value, reward, terminal, gamma)

        torch.testing.assert_allclose(expected, actual)

