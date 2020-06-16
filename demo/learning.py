import torch

def sample(chunk, batchsize):
    B = chunk.reaction.obs.rgb.shape[1]
    indices = torch.randint(B, size=(batchsize,), device=chunk.reaction.obs.rgb.device)
    batch = chunk[:, indices]
    batch = type(batch)({**batch[:-1], 'next_reaction': batch[1:].reaction})
    return batch

def gather(arr, indices):
    if isinstance(arr, dict):
        return arr.__class__({k: gather(arr[k], indices[k]) for k in arr})
    return torch.gather(arr, -1, indices.type(torch.long).unsqueeze(-1)).squeeze(-1)

def flatten(arr):
    if isinstance(arr, dict):
        return torch.cat([flatten(v) for v in arr.values()], -1)
    return arr

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

def v_trace(ratios, value, reward, reset, terminal, gamma, max_rho=1, max_c=1):
    rho = ratios.clamp(0, max_rho)
    c = ratios.clamp(0, max_c)
    dV = rho[:-1]*deltas(reward, value, reset, terminal, gamma=gamma)

    discount = (1 - reset.int())[:-1]*gamma

    v = torch.zeros_like(value)
    v[-1] = value[-1]
    for t in reversed(range(len(v)-1)):
        v[t] = value[t] + dV[t] + discount[t]*c[t]*(v[t+1] - value[t+1])

    return v.detach()

def advantages(ratios, value, reward, reset, v, gamma, max_pg_rho=1):
    rho = ratios.clamp(0, max_pg_rho)
    discount = (1 - reset.int())*gamma
    vprime = torch.cat([v[1:], value[[-1]]])
    adv = reward + discount*vprime - value
    return (rho*adv).detach()

def step(agent, opt, batch, entropy=.01, gamma=.99):
    decision = agent(batch.reaction, value=True)

    old_logits = flatten(gather(batch.decision.logits, batch.decision.actions)).sum(-1)
    new_logits = flatten(gather(decision.logits, batch.decision.actions)).sum(-1)
    ratios = (new_logits - old_logits).exp()

    reward = batch.next_reaction.reward.clamp(-1, +1)
    reset = batch.next_reaction.reset
    terminal = batch.next_reaction.terminal
    reward = reward.clamp(-1, +1)
    v = v_trace(ratios, decision.value, reward, reset, terminal, gamma=gamma)
    adv = advantages(ratios, decision.value, reward, reset, v, gamma=gamma)

    v_loss = .5*(v - decision.value).pow(2).mean() 
    p_loss = (adv*new_logits).mean()
    h_loss = -(new_logits.exp()*new_logits).mean()
    loss = v_loss - p_loss - entropy*h_loss
    
    opt.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(agent.parameters(), 40.)

    opt.step()