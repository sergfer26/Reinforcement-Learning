from .minmax_agent import MinMax_Agent
import numpy as np
import operator


class Rdm_MinMax_Agent(MinMax_Agent):

    def __init__(self):
        MinMax_Agent.__init__(self)
        self.epsilon = 0.5

    def select_action(self, key):
        state = self.key_to_state(key)
        actions = [a for a in state if a == 0]
        A_s = len(actions)
        p = self.epsilon/A_s
        bernoulli = np.random.binomial(1, p)
        if self.role == 'X':
            rewards = self.rewardsX
        else:
            rewards = self.rewardsO

        transitions = {k: v for k, v in rewards.items() if k[0] == state}

        if bernoulli != 1:
            _, a, _ = max(transitions.items(), key=operator.itemgetter(1))[0]
        else:
            a = np.random.choice(actions)
        return a


