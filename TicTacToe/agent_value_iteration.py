#!/usr/bin/env python3
import numpy as np
import collections
import json
from TicTacToe.base_agent import BaseAgent
from TicTacToe.read_tables import remap_stringkeys, remap_keys
from TicTacToe.read_tables import remap_values, remap_transits
from collections import Counter
from numpy.linalg import norm as norma

#REWARDS, TVALUES,
PLAYERS = ['X', 'O']


def create_avi():
    agent = AgentVI()
    #agent.rewards = REWARDS
    #agent.values = TVALUES
    return agent


class AgentVI(BaseAgent):
    '''
    Agente basado en el método de value iteration, 
    '''
  
    def __init__(self):
        '''
            Reward table: (source state , action, target state) -> inmediate 
            reward
            Transitions table : (state, action) -> { states_counter}
            Value table: state -> value.
        '''
        BaseAgent.__init__(self)
        self.key = 0
        self.rewards = collections.defaultdict(float)
        self.values = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.gamma = 0.5

    def calc_action_value(self, key, action):
        target_counts = self.transits[(key, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_key, count in target_counts.items():
            reward = self.rewards[(key, action, tgt_key)]
            action_value += (count / total) * (reward +
            self.gamma * self.values[tgt_key])

        return action_value

    def select_action(self, key):
        best_action, best_value, = None, None
        state = self.key_to_state(key)
        actions = [a for a, e in enumerate(state) if e == 0]
        for action in actions:
            action_value = self.calc_action_value(key, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
            else:
                pass
        return best_action

    def value_iteration(self):
        rewards = list(self.rewards.keys())
        for key, _, _ in rewards: 
            state_vals = []
            val = self.values[key]
            state = self.key_to_state(key)
            actions = [a for a, e in enumerate(state) if e == 0]
            for action in actions:
                state_vals.append(self.calc_action_value(key, action))  
            self.values[key] = max(state_vals) if len(state_vals) else val

    def get_step_info(self, key, action, reward, new_key):
        self.transits[(key, action)][new_key] += 1
        self.rewards[(key, action, new_key)] = reward
        