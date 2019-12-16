#!/usr/bin/env python3
from .base_agent import BaseAgent
from .read_tables import remap_stringkeys, remap_keys, remap_values
from .agent_mc import AgentMC
from .agent_random import RandomAgent
from .duel import duel
from .board import Board
import collections
import numpy as np
import json
import operator


class AgentMCTS(AgentMC):
    '''
    Agente basado en el método de MCTS/UTC
    '''

    def __init__(self):
        AgentMC.__init__(self)
        self.C = 1.0
        self.N = 0 # simulaciones
        self.trainer = RandomAgent()
        self.simulating = False
        self.not_leafs = [0]

    def reset_episode(self):
        '''
        Reinicia el episodio
        '''
        self.episode['states'] = []
        self.episode['actions'] = []

    def calc_ucb1(self, key):
        '''
        Calcula el valor de ucb1
        '''
        value = np.Inf
        n = self.get_total_transits(key)
        if n != 0:
            value = self.values[key] + self.C*np.sqrt(2*np.log(self.N)/n)
        return value

    def is_node_leaf(self, key):
        '''
        Determina si un estado es hoja
        '''
        if key not in self.not_leafs:
            return False
        else: 
            return True
    
    def get_other_role(self):
        '''
        Obtiene el rol del oponennte
        '''
        role = list(set(['X', 'O']) - set(self.role))
        return role[0]

    def get_next_state(self, old_key, action):
        '''
        Regresa los posibles estados, depués de que tiro el
        oponente.
        '''
        old_state = self.key_to_state(old_key)
        self.board.state = old_state
        self.board.state_to_items()
        state, _, _ = self.board.step(action, self.role)
        actions = [a for a, e in enumerate(state) if e == 0]
        states = dict()
        for a in actions:
            ns, _, _ = self.board.step(a, self.get_other_role())
            new_key = self.get_min_state(ns)[0]
            [_, ref, rots] = self.get_min_state(ns)[1]
            states[new_key] = [ref, rots]
            self.board.state = state
            self.board.state_to_items()

        self.board.reset()
        return states

    def select_action(self, key):
        '''
        Cuando no se esta simulando usa la política greedy y
        cuando se esta simulando ocupa la política aleatoria,
        en ambos casos se realiza el MCTS antes de toamar una 
        acción
        '''
        if self.simulating:
            return self.select_random_action(key)

        self.MCTS(key)
        state = self.key_to_state(key)
        actions = [a for a, e in enumerate(state) if e == 0]
        action = self.get_best_action(key, actions)

        return action

    def MCTS(self, key):
        self.selection(key)
    
    def selection(self, current):
        '''
        Fase de selección
        '''
        state = self.key_to_state(current)
        self.board.state = state
        self.board.state_to_items()
        if self.board.is_game_over() > -1:
            self.board.reset()
            return
        self.board.reset()
        actions = [a for a, e in enumerate(state) if e == 0]
        self.N += 1
        if self.is_node_leaf(current):
            value = self.calc_ucb1(current)
            if value == np.Inf:
                self.rollout(None, None, current)
            else:
                self.expansion(current)
        else:
            best_child = None
            best_value = - np.Inf
            best_action = None
            for action in actions:
                states = self.get_next_state(current, action)
                list_states = list(states.keys())
                values = list(map(lambda k: self.calc_ucb1(k), list_states))
                if not values:
                    self.reset_episode()
                    return
                if best_value < max(values):
                    index = values.index(max(values))
                    best_child = list_states[index]
                    best_action = action
                    best_value = max(values)

            self.episode['states'].append(current)
            self.episode['actions'].append(best_action)
            self.transits[(current, best_action)][best_child] += 1

            self.selection(best_child)

    def expansion(self, key):
        '''
        Fase de expansión del árbol
        '''
        state = self.key_to_state(key)
        actions = [a for a, e in enumerate(state) if e == 0]
        childs = list()
        i = 0
        for action in actions:
            states = self.get_next_state(key, action)
            list_states = list(states.keys())
            if i == 0:
                aux = states

            map(lambda nk: self.not_leafs.append(nk), list_states)
            childs += list_states
            i += 1

        action = actions[0]
        child = childs[0]
        [ref, rots] = aux[child]
        self.transits[(key, action)][child] += 1
        self.episode['states'].append(key)
        self.episode['actions'].append(action)
        self.rollout(key, action, child)

    def rollout(self, key, action, child, ref=False, rots=0):
        '''
        Es la parte de simulación
        '''
        board = Board()
        self.simulating = True
        state = self.key_to_state(child)
        board.state = state
        board.state_to_items()
        if self.role == 'X':
            duel(self, self.trainer, show=False, board=board, old_key=key, old_action=action)
        else:
            duel(self.trainer, self, show=False, board=board, old_key=key, old_action=action)

        self.simulating = False
    
    def backpropagation(self, r):
        '''
        Actualiza la información después de terminar un episodio.
        '''
        states = self.episode['states']
        actions = self.episode['actions']
        states.reverse()
        actions.reverse()
        i = 0
        for key, a in zip(states, actions):
            n = self.get_transits(key, a)
            M = self.get_total_transits(key)
            old_val = self.values[key]
            old_qval = self.qvalues[(key, a)]
            self.values[key] = ((M-1) * old_val + (self.gamma**i)*r)/M
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
