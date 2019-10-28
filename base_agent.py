#!/usr/bin/env python3
from board import Board
import collections
from collections import defaultdict
from ast import literal_eval
import numpy as np


def remap_keys(mapping, type_=float):
    dic = defaultdict(type_) 
    for k, v in mapping.items():
        dic[str(k)] = v
    return dic


def remap_stringkeys(mapping, type_=float):
    dic = defaultdict(type_)
    for k, v in mapping.items():
        k = literal_eval(k)
        dic[k] = v
    return dic


def remap_values(mapping, type_=float):
    dic = defaultdict(type_)
    for k, v in mapping.items():
        dic[k] = -v
    return dic


class BaseAgent:
    '''
    Clase abstracta que define la iterface de cualquier jugador de gato.
    Nos permite definir métodos y atributos generales del jugador.
    '''

    def __init__(self):
        '''
        Establece el tablero del jugador y su llave inicial.
        '''
        self.wins = 0
        self.board = Board()
        self.key = 0
        self.role = 'X'

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

    @staticmethod
    def state_to_matrix(state):
        '''
        Transforma el estado a una matriz.
        - Param (state): estado que es un arreglo de 1x9.
        - Regresa (matriz): matriz de 3x3.
        '''
        # -1 infers the size of the new dimension from 
        # the size of the input array
        return np.reshape(state, (-1, 3))

    @staticmethod
    def rotate_right(state):
        '''
        Rota el tablero físico 45º en sentido horario. 
        - Param (state): estado que es un arreglo de 1x9.
        - Regresa (state): estado que es un arreglo de 1x9.
        '''
        mat = BaseAgent.state_to_matrix(state)
        aux = np.zeros((3, 3), dtype=int)
        aux[:, 2] = mat[0, ]
        aux[:, 1] = mat[1, ]
        aux[:, 0] = mat[2, ]
        state = aux.reshape(-1, 9)[0, ]
        return list(state)

    @staticmethod
    def rotate_left(state):
        '''
        Rota el tablero físico 45º en sentido anti horario. 
        - Param (state): estado que es un arreglo de 1x9.
        - Regresa (state): estado que es un arreglo de 1x9.
        '''
        mat = BaseAgent.state_to_matrix(state)
        aux = np.zeros((3, 3), dtype=int)
        aux[0, ] = mat[:, 2]
        aux[1, ] = mat[:, 1]
        aux[2, ] = mat[:, 0]
        state = aux.reshape(-1, 9)[0, ]
        return list(state)

    @staticmethod
    def reflect_h(state):
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

    @staticmethod
    def reflect_v(state):
        '''
        Refleja el tablero físico de forma vertical.
        - Param (state): estado que es un arreglo de 1x9.
        - Regresa (state): estado que es un arreglo de 1x9.
        '''
        mat = BaseAgent.state_to_matrix(state)
        aux = np.zeros((3, 3), dtype=int)
        aux[:, 0] = mat[:, 2]
        aux[:, 1] = mat[:, 1]
        aux[:, 2] = mat[:, 0]
        state = aux.reshape(-1, 9)[0, ]
        return list(state)

    @staticmethod
    def state_equal(state1, state2):
        '''
        Verifica si dos estados son equivalentes bajo rotación o
        reflexión. 
        - Param (state1, state2): estados que son arreglos de 1x9.
        - Regresa (bool): condición de equivalencia
        '''
        if BaseAgent.is_rotated(state1, state2):
            return True
        else:
            state1 = BaseAgent.reflect_h(state1)
            if BaseAgent.is_rotated(state1, state2):
                return True
        return False

    @staticmethod
    def is_rotated(state1, state2):
        '''
        Verifica si dos estados son equivalentes bajo rotación.
        - Param (state1, state2): estados que son arreglos de 1x9.
        - Regresa (bool): condición de equivalencia
        '''
        for _ in range(4):
            if np.array_equal(state1, state2):
                return True
            state1 = BaseAgent.rotate_right(state1)
        return False

    @staticmethod
    def state_to_key(state):
        '''
        Transforma el vector de estados de base 3 a un número de
        base 10.
        - Param (state): estado en base 3.
        - Regresa (num): valor en base 10.
        '''
        powers = np.array(range(9))
        vec = np.array([3]*9)
        base = np.power(vec, powers)
        return int(np.dot(base, state))

    @staticmethod
    def key_to_state(num):
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
        return list(state)

    @staticmethod
    def get_min_state(state):
        '''
        Calcula todos los estados equivalentes y regresa la pareja
        (state, key) donde key es el valor mínimo de números en base 10
        de un estado y state es el acomodo asociado a dicho valor.
        - Param (state): arreglo asociado al tablero 1x9.
        - Regresa (state, key): elemtos mínimos del grupo de estados.
        '''
        states = BaseAgent.get_all_states(state)
        return min(states.items())

    @staticmethod
    def get_all_states(state):
        '''
        Obtiene todos los estados equivalentes a un estado dado
        y los pone en un diccionario donde la llave (key) es el valor
        asociado a un estado y el valor es (state) que es el arreglo a
        asociado.
        - Param (state): arreglo asociado al tablero 1x9.
        - Regresa (dict): diccionario con todos los estados.
        '''
        states = collections.defaultdict(list)
        key = BaseAgent.state_to_key(state)
        reflected = False
        states[key] = [state, reflected, 0]
        new_state = state
        for _ in range(2):
            for i in range(3):
                new_state = BaseAgent.rotate_right(new_state)
                new_key = BaseAgent.state_to_key(new_state)
                states[new_key] = [new_state, reflected, i+1]
            if not reflected:
                reflected = True
                new_state = BaseAgent.reflect_h(state)
                new_key = BaseAgent.state_to_key(new_state)
                states[new_key] = [new_state, reflected, 0]
        return states

    @staticmethod
    def get_board_action(action, ref, rots):
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
                actions = BaseAgent.reflect_h(actions)
            else:
                actions = BaseAgent.reflect_v(actions)
        for _ in range(0, rots):
            actions = BaseAgent.rotate_left(actions)
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

    def get_step_info(self, key, action, reward, new_key):
        pass
