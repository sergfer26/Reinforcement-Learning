#!/usr/bin/env python3
from base_agent import BaseAgent
from board import TicTacToe as ttt
import numpy as np

PLAYERS = ['X', 'O']

def diff(first, second):
    '''
    Obtiene la diferencia de elementos entre dos listas.
    :param first: lista de la que se obtendra la diferencia.
    :param second: lista que sacará la diferencia.
    :return: first - second
    '''
    second = set(second)
    return [item for item in first if item not in second]


class MinMax_Agent(BaseAgent):
    '''
    Define el jugador que sigue el algritmo de min max
    '''

    def __init__(self):
        '''
        Constructor
        '''
        BaseAgent.__init__(self)
        self.test_board = ttt()

    def minmax(self, state, player, mark):
        self.test_board.state = state
        self.test_board.state_to_items() 
        best = np.inf
        if player == -1:
            [mark] = diff(PLAYERS, [mark])
            best = -best

        status = self.test_board.is_game_over()
        if status > -1:
            if status == 0:
                return [None, 0]
            if status == 2:
                return [None, -1]
            else: 
                return [None, 1]

        empty_spots = [i for i, e in enumerate(state) if e == 0]
        for action in empty_spots:
            self.test_board.mark_(action, player)
            self.test_board.items_to_state()
            new_state = self.test_board.state
            [a, reward] = self.minmax(new_state, -player, mark)
            if player == -1:
                if reward > best:
                    best = reward
                    best_action = a
            else:
                if reward < best:
                    best = reward
                    best_action = a
        return [best_action, best]

    def move(self)




