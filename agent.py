#!/usr/bin/env python3
from board import TicTacToe
import collections
import numpy as np


class Agent:
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
        self.board = TicTacToe()
        self.key = 0
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)
        self.gamma = 0.1
    
    def select_random_action(self, player, state):
        '''
            Realiza un movimiento aleatorio
        '''
        while True:
            action = self.board.action_space.sample()
            if state[action] == 0:
                return action
    
    def update_dicts(self, reflected, rots, player):
        '''
            Actualiza las diccionarios asociados a
            las tablas de valores.
        '''
        state = self.base10_to_state(self.key)
        action = self.select_random_action('X', state)
        board_action = self.get_board_action(action, reflected, rots)
        new_state, reward, is_done = self.board.step(board_action, player)
        new_key = self.get_min_state(new_state)[0]
        [_, reflected, rots] = self.get_min_state(new_state)[1]
        self.rewards[(self.key, action, new_key)] = reward
        self.values[new_key] = reward
        self.transits[(self.key, action)][new_key] += 1
        return is_done, new_key, reflected, rots

    def reset_key(self):
        self.board.reset()
        return 0

    def reset_rr(self):
        return False, 0

    def play_n_random_steps(self, n):
        '''
            Realiza n turnos con jugadas aleatorias para conocer 
            estados posibles.
        '''
        self.values[self.key] = 0
        reflected = False
        rots = 0
        for _ in range(n):
            is_done, new_key, reflected, rots = self.update_dicts(reflected, rots, 'X')
            self.key = self.reset_key() if is_done else new_key
            if is_done: 
                reflected, rots = self.reset_rr()
            else:
                is_done, new_key, reflected, rots = self.update_dicts(reflected, rots, 'O')
                self.key = self.reset_key() if is_done else new_key
                if is_done:
                    reflected, rots = self.reset_rr()
        self.key = self.reset_key()

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
        return list(state)

    def rotate_left(self, state):
        '''
            Da el estado de rotar el tablero en el sentido antihorario 
            de acuerdo a las columnas del tablero. 
        '''
        mat = self.state_to_matrix(state)
        aux = np.zeros((3, 3), dtype=int)
        aux[0, ] = mat[:, 2]
        aux[1, ] = mat[:, 1]
        aux[2, ] = mat[:, 0]
        state = aux.reshape(-1, 9)[0, ]
        return list(state)


    def reflect_h(self, state):
        '''
            Intercambia el renglón 3 por el 1 (Reflexión horizontal).
        '''
        a = state[0:3]
        b = state[3:6]
        c = state[6:]
        aux = np.concatenate((b, a), axis=None)
        state = np.concatenate((c, aux), axis=None)
        return list(state)

    def reflect_v(self, state):
        '''
            Intercambia la columna 3 por la 1 (Reflexión vertical).
        '''
        mat = self.state_to_matrix(state)
        aux = np.zeros((3, 3), dtype=int)
        aux[:, 0] = mat[:, 2]
        aux[:, 1] = mat[:, 1]
        aux[:, 2] = mat[:, 0]
        state = aux.reshape(-1, 9)[0, ]
        return list(state)

    def state_equal(self, state1, state2):
        '''
        Verifica de si dos estados son equivalentes.
        '''
        if self.is_rotated(state1, state2):
            return True
        else:
            state1 = self.reflect_h(state1)
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
        n = 9 - len(digits)
        for i in range(0, n):
            digits.insert(0, 0)
        digits.reverse()
        state = np.array(digits)
        return state
 
    def get_min_state(self, state):
        '''
            Obtine el minimo valor de estado asociado.
        '''
        states = self.get_all_states(state)
        return min(states.items())

    def get_all_states(self, state):
        states = collections.defaultdict(list)
        key = self.state_to_base10(state)
        reflected = False
        states[key] = [state, reflected, 0] 
        new_state = state
        for j in range(2):
            for i in range(3):
                new_state = self.rotate_right(new_state)
                new_key = self.state_to_base10(new_state)
                states[new_key] = [new_state, reflected, i+1]
            if not reflected:
                reflected = True
                new_state = self.reflect_h(state)
                new_key = self.state_to_base10(new_state)
                states[new_key] = [new_state, reflected, 0]
        return states

    def get_board_action(self, action, reflected, rots):
        actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        if reflected:
            if rots % 2 == 0:
                actions = self.reflect_h(actions)
            else: 
                actions = self.reflect_v(actions)
        for i in range(0, rots):
            actions = self.rotate_left(actions)
        return actions.index(action)



