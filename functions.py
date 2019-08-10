import numpy as np


def state_to_matrix(state):
    '''
        pasa el estado a un 1%arreglo de arreglos
        1x9 -> 3x3 
    '''
    # -1 infers the size of the new dimension from 
    # the size of the input array
    return np.reshape(state, (-1, 3))

def rotate_right(state):
    '''
        da el estado de rotar el tablero en el sentido horario 
        de acuerdo a las columnas del tablero   
    '''
    mat = state_to_matrix(state)
    aux = np.zeros((3, 3), dtype=int)
    aux[:, 2] = mat[0, ]
    aux[:, 1] = mat[1, ]
    aux[:, 0] = mat[2, ]
    state = aux.reshape(-1, 9)[0, ]
    return state


def reflect(state):
    '''
        intercambia la columna 3 por la 1 (Reflexión de abajo horizontal)
    '''
    a = state[0:3]
    b = state[3:6]
    c = state[6:]
    aux = np.concatenate((b, a), axis=None)
    state = np.concatenate((c, aux), axis=None)
    return state

def state_equal(state1, state2):
    '''
        Verifica de si dos estados son equivalentes
    '''
    if is_rotated(state1, state2):
        return True
    else:
        state1 = reflect(state1)
        if is_rotated(state1, state2):
            return True
    return False

def is_rotated(state1, state2):
    '''
        verifica sin un estado es equivalente a otro aplicando rotaciones
    '''
    for i in range(4):
        if np.array_equal(state1, state2):
            return True
        state1 = rotate_right(state1)
    return False

def state_to_base10(state):
    '''
        transforma el vector de estados de base 3 a un número de base 10
    '''
    powers = np.array(range(9))
    vec = np.array([3]*9)
    base = np.power(vec, powers)
    return np.dot(base, state)

def base10_to_state(num):
    '''
        Pasa el número de base 10 al estado asociado
    '''
    digits = []
    base = 3
    while num > 0:
        digits.insert(0, num % base)
        num = num // base
    digits.reverse()
    state = np.array(digits)
    return state
    


