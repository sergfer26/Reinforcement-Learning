#!/usr/bin/env python3
from base_agent import BaseAgent
from board import TicTacToe as ttt

PLAYERS = ['X', 'O']

def diff(first, second):
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
        if player == -1:
            [mark] = diff(PLAYERS, [mark])

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
            action_reward = self.minmax(new_state, -player, mark)



