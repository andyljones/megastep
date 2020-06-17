import gym

class MultiDiscrete(gym.spaces.Space):

    def __init__(self, n_agents, n_actions):
        super().__init__()
        self.shape = (n_agents, n_actions)

class MultiImage(gym.spaces.Space):

    def __init__(self, n_agents, C, H, W):
        super().__init__()
        self.shape = (n_agents, C, H, W)

class MultiVector(gym.spaces.Space):

    def __init__(self, n_agents, dim):
        super().__init__()
        self.shape = (n_agents, dim)