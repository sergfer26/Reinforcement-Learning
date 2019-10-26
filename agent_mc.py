#!/usr/bin/env python3
from base_agent import BaseAgent
from agent_value_iteration import AgentVI
import collections

GAMMA = 0.5


class AgentMC(AgentVI):
    '''
    Agente basado en Montecarlo
    '''

    def __init__(self):
        AgentVI.__init__(self)
        self.transits = collections.defaultdict(int)
        self.episode = {'states': set(), 'actions': []}

    def reset_episode(self):
        self.episode['states'].clear()
        self.episode['actions'] = []
            
    def select_action(self, key):
        best_action, best_value, = None, None
        key_action = list(self.rewards.keys())
        actions = [a for k, a, nk in key_action if k == key]
        for action in actions:
            action_value = self.calc_action_value(key, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
            else:
                pass
        return best_action

    def value_update_mc(self, episode):
        states = episode['states']
        actions = episode['actions']
        states.reverse()
        actions.reverse()
        tkey = states.pop(0)
        key = states[0]
        a = actions[0]
        self.values[tkey] = self.rewards[(key, a, tkey)]
        for key, action in zip(states, actions):
            N = self.transits[key]
            old_val = self.values[key]
            val = self.calc_action_value(key, action)
            self.values[key] = ((N-1) * old_val + val)/N

        self.values[tkey] = self.rewards[(key, action, tkey)]
        self.reset_episode()

    def play_episode(self, env):
        episode = {'states': [], 'actions': []}
        env.reset()
        key = 0
        k = 0 
        reflected = False
        rots = 0
        players = ['X', 'O']
        while True:
            self.transits[key] += 1
            self.episode['states'].add(key)
            action = self.select_action(key)
            self.episode['actions'].append(action)
            board_action = self.get_board_action(action, reflected, rots)
            new_state, reward, is_done = env.step(board_action, players[k % 2])
            new_key = self.get_min_state(new_state)[0]
            [_, reflected, rots] = self.get_min_state(new_state)[1]
            self.rewards[(key, action, new_key)] = reward
            key = new_key
            if is_done:
                break
            k += 1 
        self.episode['states'].add(new_key)
        return episode

    def get_step_info(self, key, action, reward, new_key):
        self.episode['states'].add(key)
        if key != 0:
            self.episode['states'].add(new_key)
            self.episode['actions'].append(action) 


if __name__ == "__main__":
    player = AgentMC()
    from board import Board as board
    board = board()
    player.play_n_random_games(10000)
    episode = player.play_episode(board)
    print(episode)
