from .agent_tabular_qlearning import Agent_TQL


class Agent_SARSA(Agent_TQL):

    def __init__(self):
        Agent_TQL.__init__(self)
        self.SAR = {'state': 0, 'action': 0, 'reward': 0}

    def reset_SAR(self):
        self.SAR = {'state': 0, 'action': 0, 'reward': 0}

    def get_step_info(self, key, action, reward, new_key):

        if key != 0:
            na = action
            s = self.SAR['state']
            a = self.SAR['action']
            r = self.SAR['reward']
            ns = key
            self.value_update(s, a, r, ns, na)

        self.SAR['state'] = key
        self.SAR['action'] = action
        self.SAR['reward'] = reward

        if not new_key:
            self.reset_SAR()
            self.value_update(ns, na, reward, None, None)

    def value_update(self, s, a, r, ns, na):
        old_val = self.values[(s, a)]
        new_val = r + self.gamma * self.values[(ns, na)] - old_val
        self.values[(s, a)] += self.alpha * new_val
