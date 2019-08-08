import numpy as np

def board_to_matrix(board):
    '''
        pasa arreglo de numeros a arreglo de arreglos
        1x9 -> 3x3 
    '''
    # -1 infers the size of the new dimension from 
    # the size of the input array
    return np.reshape(board, (-1, 3))

def rotate_left(board):
    '''
        rota el tablero de acuerdo a las columnas del tablero   
    '''
    mat = board_to_matrix(board)
    aux = np.zeros((3, 3))
    aux[:, 2] = mat[0, ]
    aux[:, 1] = mat[1, ]
    aux[:, 0] = mat[2, ]
    board = aux.reshape(-1, 9)
    return board


def reflect_board(board):
    '''
        intercambia la columna 3 por la 1 (Reflexión de derecha a izquierda)
    '''
    mat = board_to_matrix(board)
    a = mat[:, 0]
    b = mat[:, 1]
    c = mat[:, 2]
    aux = np.concatenate((b, a), axis=None)
    board = np.concatenate((c, aux), axis=None)
    return board

def is_same_board(board1, board2):
    '''
        Verifica de si dos tableros son equivalentes
    '''
    if is_rotated(board1, board2):
        return True
    else:
        board1 = reflect_board(board1)
        if is_rotated(board1, board2):
            return True
    return False

def is_rotated(board1, board2):
    '''
        verifica sin un tablero es equivalente aplicando rotaciones
    '''
    for i in range(4):
        if np.array_equal(board1, board2):
            return True
        board1 = rotate_left(board1)
    return False
   