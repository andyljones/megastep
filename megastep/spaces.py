"""TODO-DOCS Spaces docs"""

class MultiEmpty:
    
    def __init__(self):
        pass

class MultiVector:

    def __init__(self, n_agents, dim):
        super().__init__()
        self.shape = (n_agents, dim)

class MultiImage:

    def __init__(self, n_agents, C, H, W):
        super().__init__()
        self.shape = (n_agents, C, H, W)

class MultiConstant:

    def __init__(self, n_agents):
        self.shape = (n_agents,)

class MultiDiscrete:

    def __init__(self, n_agents, n_actions):
        super().__init__()
        self.shape = (n_agents, n_actions)