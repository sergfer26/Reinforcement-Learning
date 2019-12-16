#!/usr/bin/env python3
import collections
import numpy as np
from .base_agent import BaseAgent
from .agent_value_iteration import AgentVI


class AgentMC(AgentVI):
    '''
    Agente basado en Montecarlo
    '''

    def __init__(self):
        AgentVI.__init__(self)
        self.episode = {'states': [], 'actions': []}
        self.epsilon = 0.2
        self.gamma = 0.5
        self.qvalues = collections.defaultdict(float)

    def reset_episode(self):
        '''
        Reinicia el episodio
        '''
        self.episode['states'] = []
        self.episode['actions'] = []

    def get_best_action(self, key, actions):
        '''
        Obtine la mejor acción posible dado que se esta en un estado,
        compara los valores de q.
        '''
        qvals = list(map(lambda a: self.qvalues[(key, a)], actions))
        if all(qvals == 0 for v in qvals):
            best_action = np.random.choice(actions)
        else:
            index = qvals.index(max(qvals))
            best_action = actions[index]
        return best_action

    def select_action(self, key):
        '''
        Selecciona una acción dado que se esta en un estado, 
        sigue la política e-greedy.
        '''
        state = self.key_to_state(key)
        actions = [a for a, e in enumerate(state) if e == 0]
        A_s = len(actions)
        p = self.epsilon/A_s
        bernoulli = np.random.binomial(1, p)
        if bernoulli == 1:
            action = np.random.choice(actions)
        else:
            action = self.get_best_action(key, actions)
            
        return action
    
    def get_transits(self, key, action):
        '''
        Obtiene la catidad de veces que se ha pasado por un (estado, acción)
        '''
        target_counts = self.transits[(key, action)]
        return sum(target_counts.values())

    def get_total_transits(self, key):
        '''
        Obtiene la catidad de veces que se ha pasado por un estado
        '''
        state = self.key_to_state(key)
        actions = [a for a, e in enumerate(state) if e == 0]
        transits = list(map(lambda a: self.get_transits(key, a), actions))
        return sum(transits)

    def backpropagation(self, r):
        '''
        Al llegar a los estados terminales, la función actualiza las tablas 
        con la información aprendida
        '''
        states = self.episode['states']
        actions = self.episode['actions']
        states.reverse()
        actions.reverse()
        i = 0
        for key, a in zip(states, actions):
            n = self.get_transits(key, a)
            N = self.get_total_transits(key)
            old_val = self.values[key]
            old_qval = self.qvalues[(key, a)]
            self.values[key] = ((N-1) * old_val + (self.gamma**i)*r)/N
            self.qvalues[(key, a)] = ((n-1) * old_qval + (self.gamma**i)*r)/n
            i += 1

        self.reset_episode()

    def get_step_info(self, key, action, reward, new_key):
        '''
        Obtiene información del ambiente, se usa en duels
        '''
        self.episode['states'].append(key)
        self.episode['actions'].append(action)
        self.transits[(key, action)][new_key] += 1
        if not new_key:
            self.backpropagation(reward)
            self.board.reset()
