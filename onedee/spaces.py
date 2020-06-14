import gym

class MultiDiscrete(gym.spaces.Space):

    def __init__(self, n_agents, n_actions):
        self.shape = (n_agents, n_actions)

class MultiImage(gym.spaces.Space):

    def __init__(self, n_agents, C, H, W):
        self.shape = (n_agents, C, H, W)