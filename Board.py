#!/usr/bin/env python3
import numpy as np
import gym
from gym import spaces


class TicTacToe(gym.Env): 
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(9)
        # self.observation_space = spaces.Discrete(9)
        self.items = ['']*9
        self.state = [0]*9
        self.syms = {'': 0, 'O': 2, 'X': 1}

    def reset(self):
        self.items = [''] * 9
        self.state = [0] * 9
        self.done = False
        return self.state
        

    def show_board(self):
        board = """
             {} | {} | {}
            ----------
             {} | {} | {}
            ----------
             {} | {} | {}
        """.format(*self.items)
        print(board)

    def mark_(self, action, mark):
            if self.items[action] == '':
                self.items[action] = mark
            else:
                print('espacio ocupadp')

    def items_to_state(self):
        '''
            pasa los elementos de tablero a un estado
        '''
        self.state = [self.syms.get(sym) for sym in self.items]

    def state_to_items(self):
        '''
            recibe un estado y actualiza los elemetos del tablero
        '''
        self.items = [self.syms[key] for key in self.state]

    def is_X_winner(self, x, y, z):
        if x == y == z and x == 'X':
            return 1
        elif x == y == z and x == 'O':
            return 2
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
        for i in range(0, 3):
            j = 3*i
            status = self.is_X_winner(items[i], items[i+3], items[i+6])
            if status in [1, 2]:
                return status

            status = self.is_X_winner(items[j], items[j+1], items[j+2])
            if status in [1, 2]:
                return status

        status = self.is_X_winner(items[0], items[4], items[8])
        if status in [1, 2]:
                return status

        status = self.is_X_winner(items[2], items[4], items[6])

        if status == -1 and '' not in self.items:
            return 0

        return status
        
    def step(self, action):
        self.done = False
        reward = 0.0
        self.mark_(action, 'X')
        status = self.is_game_over()
        if status >= 0:
            self.done = True
            if status == 1:
                reward = 1.0
        self.items_to_state()
        self.show_board()
        if self.done:
            pass
        else:
            while True:
                action_O = self.action_space.sample()
                if self.state[action_O] == 0:
                    break
            self.mark_(action_O, 'O')
            self.items_to_state()
            self.show_board()
        return self.state, reward, self.done
