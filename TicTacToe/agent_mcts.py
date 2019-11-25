#!/usr/bin/env python3
from .base_agent import BaseAgent
from .read_tables import remap_stringkeys, remap_keys, remap_values
from .base_agent import BaseAgent
from .agent_random import RandomAgent
from .duel import duel
import collections
import numpy as np
import json
import operator


class AgentMCTS(BaseAgent):

    def __init__(self):
        BaseAgent.__init__(self)
        self.values = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.episode = {'states': []}
        self.C = 1.0
        self.N = 0
        self.trainer = RandomAgent()
        self.training = False

    def reset_episode(self):
        self.episode['states'] = []
        self.episode['actions'] = []

    def get_total_transits(self, key):
        state = self.key_to_state(key)
        actions = [a for a, e in enumerate(state) if e == 0]
        n = 0
        for action in actions: 
            target_counts = self.transits[(key, action)]
            n += sum(target_counts.values())
        
        return n

    def calc_ucb1(self, key):
        value = np.Inf
        n = self.get_total_transits(key)
        if n != 0:
            value = self.values[key] + np.sqrt(2*np.log(self.N)/n)
        return value

    def is_node_leaf(self, key):
        if self.transits[key]:
            return False
        else: 
            return True

    def backpropagation(self, reward):
        states = self.episode['states']
        states.reverse()
        for key in states:
            n = self.get_total_transits(key)
            if n != 0:
                old_val = self.values[key]
                val = reward
                self.values[key] = ((n-1) * old_val + val)/n

        self.reset_episode()

    def rollout(self, key):
        self.training = True
        state = self.key_to_state(key)
        self.board.state = state
        self.board.state_to_items()
        if self.role == 'X':
            duel(self, self.trainer, show=False, board=self.board)
        else:
            duel(self.trainer, self, show=False, board=self.board)

        self.board.reset()
        self.training = False

    def expansion(self, key):
        state = self.key_to_state(key)
        actions = [a for a, e in enumerate(state) if e == 0]
        for action in actions:
            s, _, _ = self.board(action, self.role)
            self.board.reset()
            new_key = self.get_min_state(s)[0]
            self.transits[key][new_key] = 0
                
        childs = list(self.transits[key].keys())
        child = childs[0]
        self.rollout(child)

    def select_action(self, current):
        state = self.key_to_state(current)
        actions = [a for a, e in enumerate(state) if e == 0]

        if self.training:
            return self.select_random_action(current)

        best_action = None
        self.N += 1
        self.episode['states'].append(current)
        if self.is_node_leaf(current):
            value = self.calc_ucb1(current)
            if value == np.Inf:
                self.rollout(current)
            else: 
                self.expansion(current)
        else:
            best_child = None
            best_value = - np.Inf
            for action in actions:
                self.board.state = state
                self.board.state_to_items()
                s, _, _ = self.board(action, self.role)
                self.board.reset()
                new_key = self.get_min_state(s)[0]
                value = self.calc_ucb1(new_key)
                if best_value < value:
                    best_child = new_key
                    best_action = action
                    best_value = value

            self.transits[(current, best_action)][best_child] += 1
            self.select_action(best_child)
        if not best_action:
            best_action = actions[0]

        return best_action
    
    def get_step_info(self, key, action, reward, new_key):
        self.episode['states'].append(key)
        if self.transits[(key, action)][new_key]:
            self.transits[(key, action)][new_key] += 1
        if not new_key:
            self.backpropagation(reward)
            self.board.reset()
