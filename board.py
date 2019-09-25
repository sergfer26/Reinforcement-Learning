#!/usr/bin/env python3
import gym
from gym import spaces


class TicTacToe(gym.Env): 
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.obs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.done = False
        self.items = [''] * 9
        self.state = [0] * 9
        self.syms = {'': 0, 'O': 2, 'X': 1}

    def reset(self):
        '''
            Establece las condiciones iniciales del juego.
        '''
        self.items = [''] * 9
        self.state = [0] * 9
        self.done = False
        return self.state
        
    def show_board(self):
        '''
            Muestra el estado del juego para el usuario.
        '''
        board = """
             {} | {} | {}
            ----------
             {} | {} | {}
            ----------
             {} | {} | {}
        """.format(*self.items)
        print(board)

    def mark_(self, action, player):
        '''
            Marca alguna casilla del tabler
        '''
        if self.items[action] == '':
            self.items[action] = player
        else:
            print('espacio ocupadp')
        self.items_to_state()

    def items_to_state(self):
        '''
            Pasa los elementos de tablero a un estado.
        '''
        self.state = [self.syms[key] for key in self.items]

    def state_to_items(self):
        '''
            Recibe un estado y actualiza los elemetos del tablero.
        '''
        reversed_syms = {value: key for key, value in self.syms.items()}
        self.items = [reversed_syms[value] for value in self.state]

    def is_X_winner(self, x, y, z):
        if x == y == z:
            if x == 'X':
                return 1
            elif x == 'O':
                return 2
            else:
                return -1
        else:
            return -1

    def is_game_over(self):
        '''
            Regresa el status del juego.
            -1: El juego esta en progreso
             0: El juego quedo empatado
             1: Gana X
             2: Gana O  
        '''
        status = -1
        items = self.items

        status = self.is_X_winner(items[0], items[1], items[2])
        if status in [1, 2]:
            return status 
        
        status = self.is_X_winner(items[3], items[4], items[5])
        if status in [1, 2]:
            return status
        
        status = self.is_X_winner(items[6], items[7], items[8])
        if status in [1, 2]:
            return status

        status = self.is_X_winner(items[0], items[3], items[6])
        if status in [1, 2]:
            return status
        
        status = self.is_X_winner(items[1], items[4], items[7])
        if status in [1, 2]:
            return status
        
        status = self.is_X_winner(items[2], items[5], items[8])
        if status in [1, 2]:
            return status

        status = self.is_X_winner(items[0], items[4], items[8])
        if status in [1, 2]:
            return status

        status = self.is_X_winner(items[2], items[4], items[6])

        if status == -1 and '' not in self.items:
            return 0

        return status

    def step(self, action, player):
        '''
            Recibe una acción de un jugador y realiza el 
            movimiento
        '''
        reward = 0.0
        self.mark_(action, player)
        status = self.is_game_over()
        if status >= 0:
            self.done = True
            if status == 1:
                reward = 1.0
        print('------------ turno de ', player, '------------')
        self.show_board()
        return self.state, reward, self.done
