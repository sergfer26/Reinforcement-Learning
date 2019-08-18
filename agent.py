#!/usr/bin/env python3
from Board import TicTacToe
import gym
import collections
import numpy as np


class Agent:
  
    def __init__(self):
        '''
            Reward table: (source state , action, target state) -> inmediate 
            reward
            Transitions table : (state, action) -> { states_counter}
            Value table: state -> value.
        '''
        self.board = TicTacToe()
        self.state = self.board.state
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, n):
        '''
            Realiza n turnos con jugadas aleatorias para conocer 
            estados posibles
        '''
        for _ in range(n):
            print('jugada', _)
            while True:
                action = self.board.action_space.sample()
                if self.state[action] == 0:
                    break 

            new_state, reward, is_done = self.board.step(action)
            print(self.state_to_matrix(new_state))
            key = self.get_min_state(self.state)[0]
            new_key = self.get_min_state(new_state)[0]
            self.rewards[(key, action, new_key)] = reward
            self.transits[(key, action)][new_key] += 1
            self.state = self.board.reset() if is_done else new_state

    def state_to_matrix(self, state):
        '''
            Pasa el estado a un 1%arreglo de arreglos
            1x9 -> 3x3 .
        '''
        # -1 infers the size of the new dimension from 
        # the size of the input array
        return np.reshape(state, (-1, 3))

    def rotate_right(self, state):
        '''
            Da el estado de rotar el tablero en el sentido horario 
            de acuerdo a las columnas del tablero. 
        '''
        mat = self.state_to_matrix(state)
        aux = np.zeros((3, 3), dtype=int)
        aux[:, 2] = mat[0, ]
        aux[:, 1] = mat[1, ]
        aux[:, 0] = mat[2, ]
        state = aux.reshape(-1, 9)[0, ]
        return state

    def reflect(self, state):
        '''
            Intercambia la columna 3 por la 1 (Reflexión de abajo horizontal).
        '''
        a = state[0:3]
        b = state[3:6]
        c = state[6:]
        aux = np.concatenate((b, a), axis=None)
        state = np.concatenate((c, aux), axis=None)
        return state

    def state_equal(self, state1, state2):
        '''
        Verifica de si dos estados son equivalentes.
        '''
        if self.is_rotated(state1, state2):
            return True
        else:
            state1 = self.reflect(state1)
            if self.is_rotated(state1, state2):
                return True
        return False

    def is_rotated(self, state1, state2):
        '''
            Verifica sin un estado es equivalente a otro aplicando rotaciones.
        '''
        for i in range(4):
            if np.array_equal(state1, state2):
                return True
            state1 = self.rotate_right(state1)
        return False

    def state_to_base10(self, state):
        '''
            Transforma el vector de estados de base 3 a un número de base 10.
        '''
        powers = np.array(range(9))
        vec = np.array([3]*9)
        base = np.power(vec, powers)
        return np.dot(base, state)

    def base10_to_state(self, num):
        '''
            Pasa el número de base 10 al estado asociado.
        '''
        digits = []
        base = 3
        while num > 0:
            digits.insert(0, num % base)
            num = num // base
        digits.reverse()
        state = np.array(digits)
        return state
 
    def get_min_state(self, state):
        '''
            Obtine el minimo valor de estado asociado 
        '''
        states = collections.defaultdict(list)
        key = self.state_to_base10(state)
        states[key] = state 
        new_state = state
        for j in range(2):
            for i in range(4):
                new_state = self.rotate_right(new_state)
                new_key = self.state_to_base10(new_state)
                states[new_key] = new_state
            new_state = self.reflect(new_state)
        return min(states.items())



