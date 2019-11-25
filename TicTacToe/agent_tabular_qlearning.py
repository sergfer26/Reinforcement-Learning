#!/usr/bin/env python3
from .base_agent import BaseAgent
from .read_tables import remap_stringkeys, remap_keys, remap_values
import collections
import numpy as np
import json
import os


class Agent_TQL(BaseAgent):
    '''
    Agente basado en tabular q learning
    '''

    def __init__(self):
        BaseAgent.__init__(self)
        self.values = collections.defaultdict(float) 
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.5

    def best_value_and_action(self, key):
        action_value = 0.0
        best_value, best_action = None, None
        if key:
            state = self.key_to_state(key)
            actions = [a for a, e in enumerate(state) if e == 0]
            for action in actions:
                action_value = self.values[(key, action)]
                if best_value is None or best_value < action_value:
                    best_value = action_value
                    best_action = action
        else:
            best_value = action_value
        return best_value, best_action

    def value_update(self, k, a, r, nk):
        best_value, _ = self.best_value_and_action(nk)
        if not best_value:
            best_value = 0.0
        old_val = self.values[(k, a)]
        new_val = r + self.gamma * best_value - old_val
        self.values[(k, a)] += self.alpha * new_val

    def select_action(self, key):
        state = self.key_to_state(key)
        actions = [a for a, e in enumerate(state) if e == 0]
        A_s = len(actions)
        p = self.epsilon/A_s
        bernoulli = np.random.binomial(1, p)
        if bernoulli == 1:
            # print('Accion Aleatoria!')
            a = np.random.choice(actions)
        else:
            values = map(lambda a: self.values[(key, a)], actions)
            values = list(values)
            if all(values == 0 for v in values):
                a = np.random.choice(actions)
            else:
                index = values.index(max(values))
                a = actions[index]
        return a

    def get_step_info(self, key, action, reward, new_key):
        self.value_update(key, action, reward, new_key)
