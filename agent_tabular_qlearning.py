#!/usr/bin/env python3
from base_agent import BaseAgent
import collections


class Agent_TQL(BaseAgent):
    '''
    Agente basado en tabular q learning
    '''

    def __init__(self):
        BaseAgent.__init__(self)
        self.values = collections.defaultdict(float) 
        self.alpha = 0.5
        self.gamma = 0.5
        self.key = 0

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

    def sample_env(self, reflected, rots, player):
        state = self.key_to_state(self.key)
        action = self.select_random_action(state)
        board_action = self.get_board_action(action, reflected, rots)
        old_key = self.key
        new_state, reward, is_done = self.board.step(board_action, player)
        new_key = self.get_min_state(new_state)[0]
        [_, reflected, rots] = self.get_min_state(new_state)[1]
        self.values[(self.key, action)] = 0
        self.key = self.reset_key if is_done else new_key
        self.values[(self.key, action)] = 0
        return old_key, action, reward, new_key, reflected, rots, is_done

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
            new_val = r
        self.values[(k, a)] = old_val  + self.alpha * new_val 

    def play_episode(self, board):
        total_reward = 0.0
        key = 0
        reflected = False
        rots = 0
        self.board.reset()
        players = ['X', 'O']
        k = 0
        while True:
            _, action = self.best_value_and_action(key)
            board_action = self.get_board_action(action, reflected, rots)
            new_state, reward, is_done = board.step(board_action, players[k % 2])
            new_key = self.get_min_state(new_state)[0]
            [_, reflected, rots] = self.get_min_state(new_state)[1]
            total_reward += reward
            if is_done:
                break
            key = new_key
            k += 1
        return total_reward
