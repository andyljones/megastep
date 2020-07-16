import numpy as np
import torch
from rebar import arrdict, dotdict
import pandas as pd

__all__ = ['dataframe']

class MultiVector:

    def __init__(self, n_agents, dim):
        super().__init__()
        self.shape = (n_agents, dim)

class MultiDiscrete:

    def __init__(self, n_agents, n_actions):
        super().__init__()
        self.shape = (n_agents, n_actions)

def _dataframe(traj):
    if isinstance(traj, dict):
        return [([k] + kk, vv) for k, v in traj.items() for kk, vv in _dataframe(v)]
    if isinstance(traj, torch.Tensor):
        return [([], pd.Series([x for x in arrdict.numpyify(traj)]))]

def dataframe(traj, **kwargs):
    return pd.concat({'.'.join(k): v for k, v in _dataframe({**traj, **kwargs})}, 1)

class FSM:

    def __init__(self, n_envs, fsm, device='cuda'):
        self.n_envs = n_envs
        self.n_states = fsm.n_states
        self.device = torch.device(device)

        self._obs = fsm.obs.to(device)
        self._trans = fsm.trans.to(device)
        self._reward = fsm.reward.to(device)
        self._terminal = fsm.terminal.to(device)
        self._start = fsm.start.to(device)
        self._indices = fsm.indices
        self._names = fsm.names

        self._token = torch.full((self.n_envs,), -1, dtype=torch.long, device=device)

        self.obs_space = MultiVector(1, fsm.d_obs) if fsm.d_obs else spaces.MultiEmpty()
        self.action_space = MultiDiscrete(1, fsm.n_actions)

    def _reset(self, reset):
        if reset.any():
            n_reset = reset.sum()
            self._token[reset] = torch.distributions.Categorical(self._start).sample((n_reset,))

    def reset(self):
        self._reset(self._terminal.new_ones((self.n_envs,)))
        return arrdict.arrdict(
            obs=self._obs[self._token, None],
            reward=torch.zeros((self.n_envs,), dtype=torch.float, device=self.device),
            reset=torch.ones((self.n_envs), dtype=torch.bool, device=self.device),
            terminal=torch.ones((self.n_envs), dtype=torch.bool, device=self.device))

    def step(self, decision):
        actions = decision.actions[:, 0]
        reward = self._reward[self._token, actions]

        weights = self._trans[self._token, actions]
        self._token[:] = torch.distributions.Categorical(weights).sample()
        
        reset = self._terminal[self._token]
        self._reset(reset)

        return arrdict.arrdict(
            obs=self._obs[self._token, None],
            idx=self._token.clone(),
            reward=reward,
            reset=reset,
            terminal=reset)

    def solve(self, eps=1e-3, gamma=.99):
        value = self._reward.new_zeros((self.n_states,))
        while True:
            succ = (value[None, None, :]*self._trans).sum(-1)
            best = (self._reward + gamma*succ).max(-1)
            best.values[self._terminal] = 0
            change = value - best.values
            value = best.values
            
            if change.pow(2).mean().pow(.5) < eps:
                break
                
        return arrdict.arrdict(value=value, policy=best.indices)

    def dataframe(self, **kwargs):
        soln = self.solve(**kwargs)
        successor = self._trans[torch.arange(self.n_states, device=self.device), soln.policy].argmax(-1)
        successor = [self._names[i] for i in successor]
        df = pd.DataFrame(arrdict.numpyify(dict(
                    name=self._names,
                    obs=[tuple(f'{x:.2f}' for x in o) for o in arrdict.numpyify(self._obs)],
                    term=self._terminal,
                    start=self._start,
                    value=soln.value,
                    policy=soln.policy,
                    successor=successor,
                ))).sort_index()
        df.index.name = 'idx'
        return df

    def __repr__(self):
        s, a, _ = self._trans.shape
        return f'{type(self).__name__}({s}s{a}a)' 

    def __str__(self):
        return repr(self)


class State:

    def __init__(self, name, builder):
        self._name = name
        self._builder = builder

    def to(self, state, action=0, reward=0., weight=1.):
        action = int(action)
        self._builder._trans.append(dotdict.dotdict(
            prev=self._name, 
            action=action, 
            next=state, 
            reward=reward, 
            weight=weight))
        return self

    def state(self, *args, **kwargs):
        return self._builder.state(*args, **kwargs)

    def build(self):
        return self._builder.build()

class Builder:

    def __init__(self):
        self._obs = []
        self._trans = []

    def state(self, name, obs, start=0.):
        if isinstance(obs, (int, float, bool)):
            obs = (obs,)
        self._obs.append(dotdict.dotdict(state=name, obs=obs, start=start))
        return State(name, self)
    
    def build(self):
        states = (
            {x.state for x in self._obs} | 
            {x.prev for x in self._trans} | 
            {x.next for x in self._trans})

        actions = {x.action for x in self._trans}
        assert max(actions) == len(actions)-1, 'Action set isn\'t contiguous'
        
        indices = {s: i for i, s in enumerate(states)}
        names = np.array(list(states))

        n_states = len(states)
        n_actions = len(actions)
        (d_obs,) = {len(x.obs) for x in self._obs}

        obs = torch.full((n_states, d_obs), np.nan)
        start = torch.full((n_states,), 0.)
        for x in self._obs:
            obs[indices[x.state]] = torch.as_tensor(x.obs)
            start[indices[x.state]] = x.start

        trans = torch.full((n_states, n_actions, n_states), 0.)
        reward = torch.full((n_states, n_actions), 0.)
        for x in self._trans:
            trans[indices[x.prev], x.action, indices[x.next]] = x.weight
            reward[indices[x.prev], x.action] = x.reward
        
        terminal = (trans.sum(-1).max(-1).values == 0)

        assert start.sum() > 0, 'No start state declared'

        return dotdict.dotdict(
            obs=obs, trans=trans, reward=reward, terminal=terminal, start=start, 
            indices=indices, names=names,
            n_states=n_states, n_actions=n_actions, d_obs=d_obs)


def fsm(f):

    def init(self, n_envs=1, *args, **kwargs):
        fsm = f(*args, **kwargs)
        assert isinstance(fsm, dict), 'FSM description must be a dictionary. Did you forget to call `.build()`?'
        super(self.__class__, self).__init__(n_envs, fsm)

    name = f.__name__
    __all__.append(name)
    return type(name, (FSM,), {'__init__': init})

@fsm
def ObliviousConstantReward():
    return (Builder()
        .state('start', obs=(), start=1.)
            .to('end', reward=1.)
        .build())

@fsm
def ObliviousCyclicReward():
    return (Builder()
        .state('start', obs=0., start=1.)
            .to('middle', reward=1)
        .state('middle', obs=1.)
            .to('end', reward=0)
        .build())

@fsm
def ObliviousChain(n=2, r=1):
    assert n >= 2, 'Need the number of states to be at least 2'
    b = Builder()
    b.state(0, obs=0., start=1.).to(1, 0)
    for i in range(1, n):
        reward = (i == n-1)
        b.state(i, obs=i/n).to(i+1, 0, reward=reward)
    return b.build()

@fsm
def ObliviousCoin():
    return (Builder()
        .state('heads', obs=+1., start=1.)
            .to('end', 0, reward=+1)
        .state('tails', obs=-1., start=1.)
            .to('end', 0, reward=-1)
        .build())

@fsm
def ObliviousDelayedCoin():
    return (Builder()
        .state('heads-1', obs=+.5, start=1.)
            .to('heads-2')
        .state('heads-2', obs=+1.)
            .to('end', reward=+1)
        .state('tails-1', obs=-.5, start=1.)
            .to('tails-2')
        .state('tails-2', obs=-1.)
            .to('end', reward=-1)
        .build())

@fsm
def DelayedMatchCoin():
    return (Builder()
        .state('heads-1', obs=+1., start=1.)
            .to('heads-2', 0)
            .to('heads-2', 1)
        .state('heads-2', obs=0.)
            .to('end', 0, reward=+1)
            .to('end', 1, reward=-1)
        .state('tails-1', obs=0., start=1.)
            .to('tails-2', 0)
            .to('tails-2', 1)
        .state('tails-2', obs=-1.)
            .to('end', 0, reward=-1)
            .to('end', 1, reward=+1)
        .build())

@fsm
def MatchCoin():
    return (Builder()
        .state('heads', obs=+1., start=1.)
            .to('end', 0, reward=+1)
            .to('end', 1, reward=-1)
        .state('tails', obs=-1., start=1.)
            .to('end', 0, reward=-1)
            .to('end', 1, reward=+1)
        .build())

@fsm
def RandomChain(n=2, seed=0):
    assert n >= 2, 'Need the radius to be at least 2'
    b = Builder()
    random = np.random.RandomState(seed)
    actions = random.permutation([0, 1])
    (b.state(0, obs=0., start=1.)
        .to(0, action=actions[0])
        .to(1, action=actions[1]))
    for i in range(1, n):
        reward = (i == n-1)
        actions = random.permutation([0, 1])
        (b.state(+i, obs=+i/n)
            .to(0, action=actions[0])
            .to(i+1, action=actions[1], reward=+reward))
    return b.build()

  