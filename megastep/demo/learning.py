"""TODO-DOCS learning docs"""
import torch
import numpy as np

def batch_indices(chunk, batch_size):
    T, B = chunk.world.reset.shape
    batch_width = batch_size//T
    indices = torch.randperm(B, device=chunk.world.reset.device)
    indices = [indices[i:i+batch_width] for i in range(0, B, batch_width)]
    return indices

def gather(arr, indices):
    if isinstance(arr, dict):
        return arr.__class__({k: gather(arr[k], indices[k]) for k in arr})
    return torch.gather(arr, -1, indices.type(torch.long).unsqueeze(-1)).squeeze(-1)

def flatten(arr):
    if isinstance(arr, dict):
        return torch.cat([flatten(v) for v in arr.values()], -1)
    return arr

def assert_same_shape(ref, *arrs):
    for a in arrs:
        assert ref.shape == a.shape

def deltas(value, reward, target, reset, gamma=.99):
    reward, reset = reward[1:], reset[1:]
    regular_deltas = (reward + gamma*target[1:]) - value[:-1]
    return torch.where(reset, reward - value[:-1], regular_deltas)

def present_value(dv, finals, reset, alpha):
    assert_same_shape(dv, reset)

    reset = reset.type(torch.float)
    acc = finals
    result = torch.full_like(dv, np.nan)
    for t in np.arange(dv.shape[0])[::-1]:
        acc = dv[t] + acc*alpha*(1 - reset[t])
        result[t] = acc
    return result

def generalized_advantages(value, reward, v, reset, gamma, lambd=.97):
    assert_same_shape(value, reward, v, reset)

    dv = deltas(value, reward, v, reset, gamma=gamma)
    finals = torch.zeros_like(dv[-1])
    return torch.cat([present_value(dv, finals, reset[1:], lambd*gamma), finals[None]], 0).detach()

def reward_to_go(reward, value, reset, gamma):
    return torch.cat([present_value(reward[1:], value[-1], reset[1:], gamma), value[[-1]]], 0).detach()

def v_trace(ratios, value, reward, reset, gamma, max_rho=1, max_c=1):
    assert_same_shape(ratios, value, reward, reset)

    rho = ratios.clamp(0, max_rho)
    c = ratios.clamp(0, max_c)
    dV = rho[:-1]*deltas(value, reward, value, reset, gamma=gamma)

    discount = (1 - reset.int())[1:]*gamma

    A = value[:-1] + dV - discount*c[:-1]*value[1:]
    B = discount*c[:-1]

    v = torch.zeros_like(value)
    v[-1] = value[-1]
    for t in reversed(range(len(v)-1)):
        v[t] = A[t] + B[t]*v[t+1]

    return v.detach()

#########
# TESTS #
#########

def v_trace_ref(ratios, value, reward, reset, gamma=.99, max_rho=1, max_c=1):
    rho = ratios.clamp(0, max_rho)
    c = ratios.clamp(0, max_c)

    v = value.clone()
    for s in range(len(v)-1):
        for t in range(s, len(v)-1):
            prod_c = c[s:t].prod()
            if reset[t+1]:
                # If the next state is terminal, then the next value is zero
                dV = rho[t]*(reward[t+1] - value[t])
                v[s] += gamma**(t - s) * prod_c*dV
                break
            else:
                dV = rho[t]*(reward[t+1] + gamma*value[t+1] - value[t])
                v[s] += gamma**(t - s) * prod_c*dV
    return v

def test_v_trace():
    ratios = torch.tensor([1., 1., 1.])
    reward = torch.tensor([1., 2., 3.])
    value = torch.tensor([4., 5., 6.])
    gamma = 1.

    reset = torch.tensor([False, False, False])
    actual = v_trace(ratios, value, reward, reset, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([11., 9., 6.]))

    reset = torch.tensor([False, True, False])
    actual = v_trace(ratios, value, reward, reset, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([2., 9., 6.]))

def test_v_trace_ref():
    ratios = torch.tensor([1., 1., 1.])
    reward = torch.tensor([1., 2., 3.])
    value = torch.tensor([4., 5., 6.])
    gamma = 1.

    reset = torch.tensor([False, False, False])
    actual = v_trace_ref(ratios, value, reward, reset, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([11., 9., 6.]))

    reset = torch.tensor([False, True, False])
    actual = v_trace_ref(ratios, value, reward, reset, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([2., 9., 6.]))

def test_v_trace_equivalent(R=100, T=10):
    for _ in range(R):
        ratios = torch.rand((T,))
        value = torch.rand((T,))
        reward = torch.rand((T,))
        reset = torch.rand((T,)) > .8
        gamma = torch.rand(())

        expected = v_trace_ref(ratios, value, reward, reset, gamma)
        actual = v_trace(ratios, value, reward, reset, gamma)

        torch.testing.assert_allclose(expected, actual)

def test_reward_to_go():
    reward = torch.tensor([1., 2., 3.])
    value = torch.tensor([4., 5., 6.])
    gamma = 1.

    reset = torch.tensor([False, False, False])
    actual = reward_to_go(reward, value, reset, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([11., 9., 6.]))

    reset = torch.tensor([False, True, False])
    actual = reward_to_go(reward, value, reset, gamma)
    torch.testing.assert_allclose(actual, torch.tensor([2., 9., 6.]))

def test_generalized_advantages():
    reward = torch.tensor([1., 2., 3])
    value = torch.tensor([4., 5., 6.])
    gamma = 1.
    lambd = 1.

    reset = torch.tensor([False, False, False])
    adv = generalized_advantages(value, reward, value, reset, gamma=gamma, lambd=lambd)
    torch.testing.assert_allclose(adv, torch.tensor([7., 4., 0.]))

    reset = torch.tensor([False, True, False])
    adv = generalized_advantages(value, reward, value, reset, gamma=gamma, lambd=lambd)
    torch.testing.assert_allclose(adv, torch.tensor([-2., 4., 0.]))