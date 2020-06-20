import numpy as np
import torch
from rebar import arrdict, dotdict
from .. import spaces

__all__ = []

class FSMEnv:

    def __init__(self, n_envs, fsm, device='cuda'):
        self.n_envs = n_envs
        self.device = torch.device(device)

        self._obs = fsm.obs.to(device)
        self._trans = fsm.trans.to(device)
        self._reward = fsm.reward.to(device)
        self._terminal = fsm.terminal.to(device)
        self._start = fsm.start.to(device)
        self._indices = fsm.indices
        self._names = fsm.names

        self._token = torch.full((self.n_envs,), -1, dtype=torch.long, device=device)

        self.observation_space = spaces.MultiVector(1, fsm.d_obs) if fsm.d_obs else spaces.MultiEmpty()
        self.action_space = spaces.MultiDiscrete(1, fsm.n_actions)

    def _reset(self, reset):
        if reset.any():
            n_reset = reset.sum()
            self._token[reset] = torch.distributions.Categorical(self._start).sample((n_reset,))

    def reset(self):
        self._reset(self._terminal.new_ones((self.n_envs,)))
        return arrdict(
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

        return arrdict(
            obs=self._obs[self._token, None],
            reward=reward,
            reset=reset,
            terminal=reset)

    def __repr__(self):
        s, a, _ = self._trans.shape
        return f'{type(self).__name__}({s}s{a}a)' 

    def __str__(self):
        return repr(self)


class State:

    def __init__(self, name, builder):
        self._name = name
        self._builder = builder

    def to(self, state, action, reward=0., weight=1.):
        assert isinstance(action, int)
        self._builder._trans.append(dotdict(
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
        self._obs.append(dotdict(state=name, obs=obs, start=start))
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

        return dotdict(
            obs=obs, trans=trans, reward=reward, terminal=terminal, start=start, 
            indices=indices, names=names,
            n_states=n_states, n_actions=n_actions, d_obs=d_obs)


def fsm(f):

    def init(self, n_envs=1, *args, **kwargs):
        fsm = f(*args, **kwargs)
        super(self.__class__, self).__init__(n_envs, fsm)

    name = f.__name__
    __all__.append(name)
    return type(name, (FSMEnv,), {'__init__': init})

@fsm
def UnitReward():
    return Builder().state('start', (), 1.).to('start', 0, 1.).build()

@fsm
def Chain(n, r=1):
    assert n >= 2, 'Need the number of states to be at least 2'
    b = Builder()
    b.state(0, obs=0., start=1.).to(1, 0)
    for i in range(1, n-1):
        b.state(i, obs=i/n).to(i+1, 0)
    b.state(n-1, obs=n-1/n).to(n, 0, reward=r)
    return b.build()

@fsm
def CoinFlip(r=1.):
    return (Builder()
        .state('heads', obs=+1., start=1.).to('end', 0, reward=+r)
        .state('tails', obs=-1., start=1.).to('end', 0, reward=-r)
        .build())