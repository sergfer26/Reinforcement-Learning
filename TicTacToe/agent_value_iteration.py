#!/usr/bin/env python3
from .base_agent import BaseAgent
from .read_tables import REWARDS, TVALUES, remap_stringkeys, remap_keys
from .read_tables import remap_values
import collections
import json
from collections import Counter
import numpy as np
from numpy.linalg import norm as norma


GAMMA = 0.5


def create_avi():
    agent = AgentVI()
    agent.rewards = REWARDS
    agent.values = TVALUES
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
    
    def update_dicts(self, reflected, rots, player, k):
        '''
            Actualiza las diccionarios asociados a
            las tablas de valores.
        '''
        state = self.key_to_state(self.key)
        action = self.select_random_action(state)
        board_action = self.get_board_action(action, reflected, rots)
        new_state, reward, is_done = self.board.step(board_action, player)
        new_key = self.get_min_state(new_state)[0]
        [_, reflected, rots] = self.get_min_state(new_state)[1]
        self.rewards[(self.key, action, new_key)] = reward
        self.values[new_key] = 0
        return is_done, new_key, reflected, rots

    def play_n_random_games(self, n):
        '''
            Realiza n turnos con jugadas aleatorias para conocer 
            estados posibles.
        '''
        self.values[self.key] = 0
        ref = False
        rots = 0
        plyrs = ['X', 'O']
        for _ in range(n):
            k = 0
            while True:
                is_done, nk, ref, rots = self.update_dicts(ref, rots, plyrs[k % 2], k)
                self.key = self.reset_key() if is_done else nk
                k += 1 
                if is_done:
                    ref, rots = self.reset_rr()
                    break
            self.key = self.reset_key()

    def calc_action_value(self, key, action):
        rewards = self.rewards.keys() 
        lista = [nk for k, a, nk in rewards if k == key and a == action]
        if lista:
            tgt_key = lista[0]
        else:
            print(action)
            print(self.key_to_state(key))

        reward = self.rewards[(key, action, tgt_key)]
        action_value = reward + GAMMA * self.values[tgt_key]
        return action_value

    def select_action(self, key):
        best_action, best_value, = None, None
        rewards = self.rewards.keys() 
        actions = [a for k, a, nk in rewards if key == k]
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
            actions = [a for k, a, nk in rewards if key == k]
            for action in actions:
                state_vals.append(self.calc_action_value(key, action))  
            self.values[key] = max(state_vals) if len(state_vals) else val

    def set_role(self, role):
        self.role = role
        self.reawards = remap_values(self.rewards)
        self.values = remap_values(self.values)


if __name__ == "__main__":

    player = AgentVI()
    matrix = []
    player.play_n_random_games(10000)
    json.dump(player.values, open("tables/values.txt", 'w'))
    dim = len(player.values)
    y = np.repeat(0, dim)
    epsilon = 0.001
    i = 0
    while True:
        player.value_iteration()
        x = np.array(list(player.values.values()))
        matrix.append(x)
        i += 1
        if norma(x-y) < epsilon:
            break
        y = x
    matrix = np.flip(matrix, axis=0)
    matrix.reshape((i, dim))
    rewards = remap_keys(player.rewards)
    json.dump(player.values, open("tables/trained_values.txt", 'w'))
    json.dump(rewards, open("tables/rewards.txt", 'w'))
    # d2 = json.load(open("text.txt"))
