#!/usr/bin/env python3
import collections
import numpy as np
from .base_agent import BaseAgent
from .agent_value_iteration import AgentVI
from .read_tables import REWARDS, VALUES, remap_values


def create_amc():
    '''
    Crea una instancia de AgentMC con los REWARDS y VALUES
    ya identificados.
    Regresa (agent): instancia
    '''
    agent = AgentMC()
    agent.rewards = REWARDS
    agent.values = VALUES
    return agent


class AgentMC(AgentVI):
    '''
    Agente basado en Montecarlo
    '''

    def __init__(self):
        AgentVI.__init__(self)
        self.transits = collections.defaultdict(int)
        self.episode = {'states': [], 'actions': []}
        self.self.epsilon = 0.2
        self.gamma = 0.5

    def reset_episode(self):
        self.episode['states'] = []
        self.episode['actions'] = []
    
    def select_action(self, key):
        key_action = list(self.rewards.keys())
        actions = [a for k, a, nk in key_action if k == key]
        A_s = len(actions)
        p = self.epsilon/A_s
        bernoulli = np.random.binomial(1, p)
        if bernoulli == 1:
            action = np.random.choice(actions)
            return action
        else:
            values = map(lambda a: self.calc_action_value(key, a), actions) 
            values = list(values)
            if all(values == 0 for v in values):
                best_action = np.random.choice(actions)
            else:
                index = values.index(max(values))
                best_action = actions[index]
            return best_action

    def value_update_mc(self):
        states = self.episode['states']
        actions = self.episode['actions']
        states.reverse()
        actions.reverse()
        tkey = states.pop(0)
        key = states[0]
        a = actions[0]
        self.values[tkey] = self.rewards[(key, a, tkey)]
        for key, action in zip(states, actions):
            self.transits[key] += 1
            N = self.transits[key]
            old_val = self.values[key]
            val = self.calc_action_value(key, action)
            self.values[key] = ((N-1) * old_val + val)/N

        self.values[tkey] = self.rewards[(key, action, tkey)]
        self.reset_episode()

    def get_step_info(self, key, action, reward, new_key):
        self.episode['states'].append(key)
        self.episode['actions'].append(action) 
        state = self.key_to_state(new_key)
        self.board.state = state
        self.board.state_to_items()
        if self.board.is_game_over() > -1:
            self.episode['states'].append(new_key)
            self.value_update_mc()
            self.board.reset()


if __name__ == "__main__":
    player = AgentMC()
    from board import Board as board
    board = board()
    player.play_n_random_games(10000)
    episode = player.play_episode(board)
    print(episode)
