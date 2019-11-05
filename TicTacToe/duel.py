#!/usr/bin/env python3
from .board import Board as board
from .human import Human as human
import json
import numpy as np
import pandas as pd


board = board()
PLAYERS = ['X', 'O']


def duel(agent_X, agent_O, show=False):
    board.reset()
    done = False
    i = 0
    key = 0
    ref = False
    rots = 0
    while not done:
        player = PLAYERS[i % 2]
        if i % 2 == 0:
            agent = agent_X
        elif i % 2 != 0:
            agent = agent_O

        if show:
            board.show_board()
        if isinstance(agent, human):
            action = agent.select_action(key, ref, rots)
            board_action = action
        else:
            action = agent.select_action(key)
            board_action = agent.get_board_action(action, ref, rots)

        new_state, reward, done = board.step(board_action, player)
        new_key = agent.get_min_state(new_state)[0]
        [_, ref, rots] = agent.get_min_state(new_state)[1]
        i += 1
        agent_X.get_step_info(key, action, reward, new_key)
        agent_O.get_step_info(key, action, reward, new_key)
        key = new_key
    
    if show:
        board.show_board()

    return board.is_game_over()


def play_n_duels(games, agent1, agent2, show=False):
    playerX = agent1
    playerO = agent2
    prob = []
    draws = 0
    for k in range(1, games+1):
        winner_value = duel(playerX, playerO, show)
        if winner_value == 2:
            playerO.wins += 1
            aux = playerX
            playerX = playerO
            playerO = aux
            playerX.set_role('X')
            playerO.set_role('O')
        elif winner_value == 1:
            playerX.wins += 1 
        elif winner_value == 0:
            draws += 1 

        if k % 100 == 0:
            prob.append(agent2.wins/k)

    return np.array(prob)
