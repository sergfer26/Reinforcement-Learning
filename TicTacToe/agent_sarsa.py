from .agent_tabular_qlearning import Agent_TQL
from .read_tables import QVALUES

ALPHA = 0.5
GAMMA = 0.5
EPSILON = 0.4


class Agent_SARSA(Agent_TQL):

    def __init__(self):
        BaseAgent.__init__(self)
        self.SARSA = {'actions': [], 'states': [], 'reward': 0}

    def reset_SARSA(self):
        self.SARSA = {'actions': [], 'states': [], 'reward': 0}

    def get_step_info(self, key, action, reward, new_key):
        if self.role == 'X':
            pass
        else: 
            reward = -reward
        if 
            self.value_update(key, action, reward, new_key, new_action)

    def value_update(self, s, a, r, ns, na):
        old_val = self.values[(s, a)]
        if na:
            new_val = r + GAMMA * self.values[(ns, na)] - old_val
        else:
            new_val = r - old_val
        self.values[(s, a)] = old_val + ALPHA * new_val
