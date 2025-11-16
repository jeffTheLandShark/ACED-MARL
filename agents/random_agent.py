import numpy as np


class RandomAgent:
    """Returns random actions for the environment.

    Use `act(observations, readiness_mask)` to select actions.
    If readiness_mask is None, all agents are allowed to act.
    """

    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def act(self, observations: np.ndarray, readiness_mask: np.ndarray | None = None):
        # action space: 0..5 inclusive -> we will return random ints in that range
        if readiness_mask is None:
            readiness_mask = np.ones(self.n_agents, dtype=bool)
        actions = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            if readiness_mask[i]:
                actions[i] = np.random.randint(0, 6)
            else:
                actions[i] = 0
        return actions
