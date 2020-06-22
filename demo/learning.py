import torch
import numpy as np

def sample(chunk, batchsize):
    B = chunk.world.reward.shape[1]
    indices = torch.randint(B, size=(batchsize,), device=chunk.world.reward.device)
    batch = chunk[:, indices]
    return batch

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

def v_trace(ratios, value, reward, reset, terminal, gamma, max_rho=1, max_c=1):
    rho = ratios.clamp(0, max_rho)
    c = ratios.clamp(0, max_c)
    dV = rho[:-1]*deltas(value, reward, value, reset, terminal, gamma=gamma)

    discount = (1 - reset.int())[1:]*gamma

    A = value[:-1] + dV - discount*c[:-1]*value[1:]
    B = discount*c[:-1]

    v = torch.zeros_like(value)
    v[-1] = value[-1]
    for t in reversed(range(len(v)-1)):
        v[t] = A[t] + B[t]*v[t+1]

    return v.detach()

def reward_to_go(value, reward, reset, terminal, gamma):
    terminated = torch.where(reset & ~terminal, value, reward)
    return torch.cat([present_value(terminated[:-1], value[-1], reset, gamma), value[[-1]]], 0).detach()

def advantages(ratios, valuez, rewardz, vz, reset, terminal, gamma, max_pg_rho=1):
    rho = ratios.clamp(0, max_pg_rho)
    return (rho[:-1]*deltas(valuez, rewardz, vz, reset, terminal, gamma=gamma)).detach()

def generalized_advantages(value, reward, v, reset, terminal, gamma, lambd=.97):
    dV = deltas(value, reward, v, reset, terminal, gamma=gamma)
    return present_value(dV, torch.zeros_like(dV[-1]), reset, lambd*gamma).detach()

def explicit_v_trace(ratios, value, reward, reset, terminal, gamma=.99, max_rho=1, max_c=1):
    rho = ratios.clamp(0, max_rho)
    c = ratios.clamp(0, max_c)

    v = value.clone()
    for s in range(len(v)-1):
        for t in range(s, len(v)-1):
            prod_c = c[s:t].prod()
            if terminal[t+1]:
                # If the next state is terminal, then the next value is zero
                dV = rho[t]*(reward[t+1] - value[t])
                v[s] += gamma**(t - s) * prod_c*dV
                break
            elif reset[t+1]:
                # If the next state is a reset, assume the next value would've 
                # explained the intervening reward perfectly
                break
            else:
                dV = rho[t]*(reward[t+1] + gamma*value[t+1] - value[t])
                v[s] += gamma**(t - s) * prod_c*dV
    
    return v

def test_v_trace_trivial():
    ratios = torch.tensor([1., 1.])
    reward = torch.tensor([1., 2.])
    value = torch.tensor([3., 4.])
    reset = torch.tensor([False, False])
    terminal = torch.tensor([False, False])
    gamma = 1.

    expected = explicit_v_trace(ratios, value, reward, reset, terminal, gamma)
    torch.testing.assert_allclose(expected, torch.tensor([6., 4.]))

    actual = v_trace(ratios, value, reward, reset, terminal, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([6., 4.]))

def test_v_trace_reset():
    ratios = torch.tensor([1., 1.])
    reward = torch.tensor([1., 2.])
    value = torch.tensor([3., 4.])
    reset = torch.tensor([False, True])
    terminal = torch.tensor([False, False])
    gamma = 1.

    expected = explicit_v_trace(ratios, value, reward, reset, terminal, gamma)
    torch.testing.assert_allclose(expected, torch.tensor([3., 4.]))

    actual = v_trace(ratios, value, reward, reset, terminal, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([3., 4.]))

def test_v_trace_terminal():
    ratios = torch.tensor([1., 1.])
    reward = torch.tensor([1., 2.])
    value = torch.tensor([3., 4.])
    reset = torch.tensor([False, True])
    terminal = torch.tensor([False, True])
    gamma = 1.

    expected = explicit_v_trace(ratios, value, reward, reset, terminal, gamma)
    torch.testing.assert_allclose(expected, torch.tensor([2., 4.]))

    actual = v_trace(ratios, value, reward, reset, terminal, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([2., 4.]))

def test_v_trace_random(R=100, T=10):
    for _ in range(R):
        ratios = torch.rand((T,))
        value = torch.rand((T,))
        reward = torch.rand((T,))
        reset = torch.rand((T,)) > .8
        terminal = reset & (torch.rand((T,)) > .5)
        gamma = torch.rand(())

        expected = explicit_v_trace(ratios, value, reward, reset, terminal, gamma)
        actual = v_trace(ratios, value, reward, reset, terminal, gamma)

        torch.testing.assert_allclose(expected, actual)

def test_advantages_trivial():
    ratios = torch.tensor([1., 1.])
    reward = torch.tensor([1., 2.])
    value = torch.tensor([3., 4.])
    reset = torch.tensor([False, False])
    terminal = torch.tensor([False, False])
    gamma = 1.

    adv = advantages(ratios, value, reward, value, reset, terminal, gamma=gamma)
    torch.testing.assert_allclose(adv, torch.tensor([3.]))

def test_advantages_reset():
    ratios = torch.tensor([1., 1.])
    reward = torch.tensor([1., 2.])
    value = torch.tensor([3., 4.])
    reset = torch.tensor([False, True])
    terminal = torch.tensor([False, False])
    gamma = 1.

    adv = advantages(ratios, value, reward, value, reset, terminal, gamma=gamma)
    torch.testing.assert_allclose(adv, torch.tensor([0.]))

def test_advantages_terminal():
    ratios = torch.tensor([1., 1.])
    reward = torch.tensor([1., 2.])
    value = torch.tensor([3., 4.])
    reset = torch.tensor([False, True])
    terminal = torch.tensor([False, True])
    gamma = 1.

    adv = advantages(ratios, value, reward, value, reset, terminal, gamma=gamma)
    torch.testing.assert_allclose(adv, torch.tensor([-1.]))