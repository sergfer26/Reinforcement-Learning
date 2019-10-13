#!/usr/bin/env python3
from board import TicTacToe
import collections
import numpy as np


class BaseAgent:
    '''
    Clase abstracta que define la iterface de cualquier jugador de gato.
    Nos permite definir métodos y atributos generales del jugador.
    '''

    def __init__(self):
        '''
        Establece el tablero del jugador y su llave inicial.
        '''
        self.board = TicTacToe()
        self.key = 0

    def select_random_action(self, state):
        '''
        Selecciona una acción aleatoria legal en el tablero.
        - Param (state): estado/acomodo de un tablero.
        - Regresa (action): acción/posición que se tomo. 
        '''
        while True:
            action = self.board.action_space.sample()
            if state[action] == 0:
                return action

    def state_to_matrix(self, state):
        '''
        Transforma el estado a una matriz.
        - Param (state): estado que es un arreglo de 1x9.
        - Regresa (matriz): matriz de 3x3.
        '''
        # -1 infers the size of the new dimension from 
        # the size of the input array
        return np.reshape(state, (-1, 3))

    def rotate_right(self, state):
        '''
        Rota el tablero físico 45º en sentido horario. 
        - Param (state): estado que es un arreglo de 1x9.
        - Regresa (state): estado que es un arreglo de 1x9.
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
        Rota el tablero físico 45º en sentido anti horario. 
        - Param (state): estado que es un arreglo de 1x9.
        - Regresa (state): estado que es un arreglo de 1x9.
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
        Refleja el tablero físico de forma horizontal. 
        - Param (state): estado que es un arreglo de 1x9.
        - Regresa (state): estado que es un arreglo de 1x9.
        '''
        a = state[0:3]
        b = state[3:6]
        c = state[6:]
        aux = np.concatenate((b, a), axis=None)
        state = np.concatenate((c, aux), axis=None)
        return list(state)

    def reflect_v(self, state):
        '''
        Refleja el tablero físico de forma vertical. 
        - Param (state): estado que es un arreglo de 1x9.
        - Regresa (state): estado que es un arreglo de 1x9.
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
        Verifica si dos estados son equivalentes bajo rotación o
        reflexión. 
        - Param (state1, state2): estados que son arreglos de 1x9.
        - Regresa (bool): condición de equivalencia
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
        Verifica si dos estados son equivalentes bajo rotación. 
        - Param (state1, state2): estados que son arreglos de 1x9.
        - Regresa (bool): condición de equivalencia
        '''
        for i in range(4):
            if np.array_equal(state1, state2):
                return True
            state1 = self.rotate_right(state1)
        return False

    def state_to_base10(self, state):
        '''
        Transforma el vector de estados de base 3 a un número de 
        base 10.
        - Param (state): estado en base 3.
        - Regresa (num): valor en base 10.
        '''
        powers = np.array(range(9))
        vec = np.array([3]*9)
        base = np.power(vec, powers)
        return np.dot(base, state)

    def base10_to_state(self, num):
        '''
        Transforma un número de base 10 a uno en base 3.
        - Param (num): número en base 10.
        - Regresa (state): número en base 3.
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
        Calcula todos los estados equivalentes y regresa la pareja
        (state, key) donde key es el valor mínimo de números en base 10
        de un estado y state es el acomodo asociado a dicho valor.
        - Param (state): arreglo asociado al tablero 1x9.
        - Regresa (state, key): elemtos mínimos del grupo de estados.
        '''
        states = self.get_all_states(state)
        return min(states.items())

    def get_all_states(self, state):
        '''
        Obtiene todos los estados equivalentes a un estado dado
        y los pone en un diccionario donde la llave (key) es el valor
        asociado a un estado y el valor es (state) que es el arreglo a
        asociado.
        - Param (state): arreglo asociado al tablero 1x9.
        - Regresa (dict): diccionario con todos los estados.
        '''
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

    def get_board_action(self, action, ref, rots):
        '''
        Obtiene la acción de cualqueir juego a partir del juego mínimo,
        considera las rotaciones y reflexión del juego real.
        - Param 1 (action): acción/posición del juego mínimo.
        - Param 2 (ref): booleano que indica si es un tablero reflejado.
        - Param 3 (rots): indica cuantas rotaciones tiene el tablero.
        - Regresa (action): acción/posición del juego real.
        '''
        actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        if ref:
            if rots % 2 == 0:
                actions = self.reflect_h(actions)
            else:
                actions = self.reflect_v(actions)
        for _ in range(0, rots):
            actions = self.rotate_left(actions)
        return actions.index(action)

    def reset_key(self):
        '''
        Establece las condiciones iniciales del tablero y
        llave del jugador.
        - Regresa (key): llave asociada al acomodo vacío del tablero.
        '''
        self.board.reset()
        return 0

    def reset_rr(self):
        '''
        Establece la reflexión y rotaciones del tablero asociado al juego
        mínimo.
        - Regresa (ref, rots): reflexión y rotaciones.
        '''
        return False, 0
