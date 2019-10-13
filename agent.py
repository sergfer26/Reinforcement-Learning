#!/usr/bin/env python3
from base_agent import BaseAgent
import collections


class AgentVI(BaseAgent):
    '''
        Agente base, sigue la política aleatoria
    '''
  
    def __init__(self):
        '''
            Reward table: (source state , action, target state) -> inmediate 
            reward
            Transitions table : (state, action) -> { states_counter}
            Value table: state -> value.
        '''
        BaseAgent.__init__(self)
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)
        self.turns = collections.defaultdict(float)
        self.gamma = 0.5
    
    def update_dicts(self, reflected, rots, player, k):
        '''
            Actualiza las diccionarios asociados a
            las tablas de valores.
        '''
        state = self.base10_to_state(self.key)
        action = self.select_random_action(state)
        board_action = self.get_board_action(action, reflected, rots)
        new_state, reward, is_done = self.board.step(board_action, player)
        new_key = self.get_min_state(new_state)[0]
        [_, reflected, rots] = self.get_min_state(new_state)[1]
        self.rewards[(self.key, action, new_key)] = reward
        self.values[new_key] = 0
        self.turns[new_key] = k
        self.transits[(self.key, action)][new_key] += 1
        return is_done, new_key, reflected, rots

    def play_n_random_games(self, n):
        '''
            Realiza n turnos con jugadas aleatorias para conocer 
            estados posibles.
        '''
        self.values[self.key] = 0
        self.turns[self.key] = 0
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
        transits = self.transits[(key, action)]
        tgt_key = sum(transits.keys())
        count = sum(transits.values())
        reward = self.rewards[(key, action, tgt_key)]
        action_value = reward + self.gamma * self.values[tgt_key]
        return action_value

    def select_action(self, key):
        best_action, best_value, = None, None
        transits = list(self.transits.keys())
        actions = [a for k, a in transits if key == k]
        for action in actions:
            action_value = self.calc_action_value(key, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
            else:
                pass
            return best_action

    def play_episode(self, env):
        total_reward = 0.0
        env.reset()
        key = 0
        k = 0 
        reflected = False
        rots = 0
        players = ['X', 'O']
        while True:
            action = self.select_action(key)
            board_action = self.get_board_action(action, reflected, rots)
            new_state, reward, is_done = env.step(board_action, players[k % 2])
            new_key = self.get_min_state(new_state)[0]
            [_, reflected, rots] = self.get_min_state(new_state)[1]
            self.rewards[(key, action, new_key)] = reward
            self.transits[(key, action)][new_key] += 1
            total_reward += reward
            if is_done:
                break
            key = new_key
            k += 1 
        return total_reward

    def check_mark(self, action, state):
        if state[action] == 0:
            return True
        return False

    def value_iteration(self):
        transits = list(self.transits.keys())
        for key in list(self.values.keys()): 
            state_vals = []
            val = self.values[key]
            actions = [a for k, a in transits if key == k]
            for action in actions:
                state_vals.append(self.calc_action_value(key, action))  
            self.values[key] = max(state_vals) if len(state_vals) else val
    
    def get_sorted_values(self):
        turns = self.turns
        vals = self.values
        sorted_items = sorted(vals.items(), key=lambda x: turns.get(x[0])) 
        return dict(sorted_items)
        
    def values_to_list(self):
        values = self.get_sorted_values().values()
        return list(values)
