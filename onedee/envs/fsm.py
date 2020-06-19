import torch
from rebar import arrdict

class FSMEnv:

    def __init__(self, states, n_envs, device='cuda'):
        indices = {n: i for i, n in enumerate(states)}
        (d_obs,) = {len(o) for t, o, ars in states.values()}
        (n_actions,) = {len(ars) for t, o, ars in states.values()}

        self.n_envs = n_envs
        self.n_agents = 1
        self.device = torch.device(device)
        self._token = torch.zeros(n_envs, dtype=torch.int)

        term, obs, trans, reward = [], [], [], []
        for t, o, ars in states.values():
            term.append(t)
            obs.append(o)
            trans.append([indices[s] for s, r in ars])
            reward.append([r for s, r in ars])
        self._term = torch.as_tensor(term, dtype=torch.bool, device=self.device)
        self._obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
        self._trans = torch.as_tensor(trans, dtype=torch.int, device=self.device)
        self._reward = torch.as_tensor(reward, dtype=torch.float, device=self.device)

    def reset(self):
        self._token[:] = 0
        return arrdict(
            obs=self._obs[self._token, None],
            reward=torch.zeros((self.n_envs,), dtype=torch.bool, device=self.device),
            reset=torch.ones((self.n_envs), dtype=torch.bool, device=self.device),
            terminal=torch.ones((self.n_envs), dtype=torch.bool, device=self.device))

    def step(self, decision):
        reward = self._trans[self._token, decision.action]
        self._token = self._trans[self._token, decision.action]
        
        reset = self._terminal[self._token]
        self._token[reset] = 0

        return arrdict(
            obs=self._obs[self._token, None],
            reward=reward,
            reset=reset,
            terminal=reset)

def add_fsm(name, states):

    def init(self, *args, **kwargs):
        super(self.__class__, self).__init__(states, *args, **kwargs)

    globals()[name] = type(name, (FSMEnv,), {'__init__': init})

add_fsm('TwoState', {
    'start': (False, (), [('terminal', 0.)]),
    'terminal': (True, (), [('terminal', 0.)])})

