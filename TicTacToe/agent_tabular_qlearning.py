#!/usr/bin/env python3
from .base_agent import BaseAgent
from .read_tables import QVALUES, remap_stringkeys, remap_keys, remap_values
import collections
import numpy as np
from numpy.linalg import norm as norma
import json
import os


def create_atql():
    agent = Agent_TQL()
    agent.values = QVALUES
    return agent


class Agent_TQL(BaseAgent):
    '''
    Agente basado en tabular q learning
    '''

    def __init__(self):
        BaseAgent.__init__(self)
        self.values = collections.defaultdict(float) 
        self.self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.5

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
        self.values[(self.key, action)] = 0
        return is_done, new_key, reflected, rots

    def play_n_random_games(self, n):
        '''
            Realiza n turnos con jugadas aleatorias para conocer
            estados posibles.
        '''
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

    def best_value_and_action(self, key):
        best_value, best_action = None, None
        values = list(self.values.keys())
        actions = [a for k, a in values if key == k]
        for action in actions:
            action_value = self.values[(key, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, k, a, r, nk):
        best_value, _ = self.best_value_and_action(nk)
        old_val = self.values[(k, a)]
        if best_value:
            new_val = r + self.gamma * best_value - old_val
        else: 
            new_val = r - old_val
        self.values[(k, a)] = old_val + self.alpha * new_val

    def select_action(self, key):
        values = list(self.values.keys())
        actions = [a for k, a in values if key == k]
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
        if self.role == 'X':
            pass
        else: 
            reward = -reward
        self.value_update(key, action, reward, new_key)

    def set_role(self, role):
        if role != self.role:
            self.role = role
            self.values = remap_values(self.values)


if __name__ == '__main__':
    player = Agent_TQL()
    ref = False
    rots = 0
    players = ['X', 'O']
    player.values = QVALUES
    dim = len(player.values)
    y = np.repeat(0, dim)
    self.epsilon = 0.001
    i = 0
    while True:
        k, a, r, nk, ref, rots, done = player.sample_env(ref, rots, players[i % 2])
        player.value_update(k, a, r, nk)
        if done:
            print('final')
            print(player.values[(k, a)])
            player.key = 0
            k = 0
            nk = 0
            player.board.reset()
        x = np.array(list(player.values.values()))
        i += 1
        if i > 10000 and norma(x-y) < self.epsilon:
            break
        y = x
    print(i)

    values = remap_keys(player.values)
    json.dump(values, open("trained/qvalues.txt", 'w'))
