import torch
import numpy as np

def batch_indices(n_envs, batch_width, device='cuda'):
    indices = torch.randperm(n_envs, device=device)
    indices = [indices[i:i+batch_width] for i in range(0, n_envs, batch_width)]
    return indices

def gather(arr, indices):
    if isinstance(arr, dict):
        return arr.__class__({k: gather(arr[k], indices[k]) for k in arr})
    return torch.gather(arr, -1, indices.type(torch.long).unsqueeze(-1)).squeeze(-1)

def flatten(arr):
    if isinstance(arr, dict):
        return torch.cat([flatten(v) for v in arr.values()], -1)
    return arr

def deltas(value, reward, target, reset, terminal, gamma=.99):
    # Value comes from the decision before the world
    # Reward, reset and terminal come from the world to the decision
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
    reward, reset, terminal = reward[1:], reset[1:], terminal[1:]
    regular_deltas = (reward + gamma*target[1:]) - value[:-1]
    terminated_deltas = torch.where(terminal, reward - value[:-1], regular_deltas)
    return torch.where(reset & ~terminal, torch.zeros_like(reward), terminated_deltas)

def present_value(vals, finals, reset, alpha):
    reset = reset.type(torch.float)
    acc = finals
    result = torch.full_like(vals, np.nan)
    for t in np.arange(vals.shape[0])[::-1]:
        acc = vals[t] + acc*alpha*(1 - reset[t])
        result[t] = acc
    return result

def reward_to_go(reward, value, reset, terminal, gamma):
    terminated = torch.where(reset[1:] & ~terminal[1:], value[:-1], reward[1:])
    return torch.cat([present_value(terminated, value[-1], reset[1:], gamma), value[[-1]]], 0).detach()

def generalized_advantages(value, reward, v, reset, terminal, gamma, lambd=.97):
    dV = deltas(value, reward, v, reset, terminal, gamma=gamma)
    return present_value(dV, torch.zeros_like(dV[-1]), reset, lambd*gamma).detach()

def test_reward_to_go():
    reward = torch.tensor([1., 2.])
    value = torch.tensor([3., 4.])
    gamma = 1.

    reset = torch.tensor([False, False])
    terminal = torch.tensor([False, False])
    actual = reward_to_go(reward, value, reset, terminal, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([6., 4.]))

    reset = torch.tensor([False, True])
    terminal = torch.tensor([False, False])
    expected = reward_to_go(reward, value, reset, terminal, gamma)
    torch.testing.assert_allclose(expected, torch.tensor([3., 4.]))

    reset = torch.tensor([False, True])
    terminal = torch.tensor([False, True])
    expected = reward_to_go(reward, value, reset, terminal, gamma)
    torch.testing.assert_allclose(expected, torch.tensor([2., 4.]))

def test_advantages():
    reward = torch.tensor([1., 2.])
    value = torch.tensor([3., 4.])
    gamma = 1.

    reset = torch.tensor([False, False])
    terminal = torch.tensor([False, False])
    adv = generalized_advantages(value, reward, value, reset, terminal, gamma=gamma)
    torch.testing.assert_allclose(adv, torch.tensor([3.]))

    reset = torch.tensor([False, True])
    terminal = torch.tensor([False, False])
    adv = generalized_advantages(value, reward, value, reset, terminal, gamma=gamma)
    torch.testing.assert_allclose(adv, torch.tensor([0.]))

    reset = torch.tensor([False, True])
    terminal = torch.tensor([False, True])
    adv = generalized_advantages(value, reward, value, reset, terminal, gamma=gamma)
    torch.testing.assert_allclose(adv, torch.tensor([-1.]))