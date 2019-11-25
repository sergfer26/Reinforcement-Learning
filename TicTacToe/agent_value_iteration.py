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
    Agente base, 
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

    def play_random_episode(self, show=False):
        self.board.reset()
        done = False
        i = 0
        ref = False
        rots = 0
        self.values[self.key] = 0
        old_action = None
        old_key = None

        while not done:
            self.set_role(PLAYERS[i % 2])

            if show:
                self.board.show_board()
                
            action = self.select_random_action(self.key)
            board_action = self.get_board_action(action, ref, rots)
            new_state, reward, done = self.board.step(board_action, self.role)
            new_key = self.get_min_state(new_state)[0]
            [_, ref, rots] = self.get_min_state(new_state)[1]
            i += 1

            if self.role == 'X':
                reward = - reward

            if self.key != 0:
                self.rewards[(old_key, old_action, new_key)] = reward
                self.transits[(old_key, old_action)][new_key] += 1

            reward = -reward
            self.values[new_key] = 0
            old_key = self.key
            old_action = action
            self.key = new_key

        self.rewards[(old_key, old_action, None)] = reward
        self.transits[(old_key, old_action)][None] += 1
        self.board.reset()
        self.key = self.reset_key()

    def play_n_random_games(self, n):
        for _ in range(n):
            self.play_random_episode()

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
        