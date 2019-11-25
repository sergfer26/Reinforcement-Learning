#!/usr/bin/env python3
from .board import Board as board
from .human import Human as human
import json
import numpy as np
import pandas as pd


def duel(playerX, playerO, show=False, board=board()):
    playerX.role = 'X'
    playerO.role = 'O'
    done = False
    i = 0
    key = 0
    ref = False
    rots = 0
    old_key = None
    old_action = None
    player = playerX
    while True:
        if show:
            board.show_board()
        if isinstance(player, human):
            action = player.select_action(key, ref, rots)
            board_action = action
        else:
            action = player.select_action(key)
            board_action = player.get_board_action(action, ref, rots)

        new_state, reward, done = board.step(board_action, player.role)
        new_key = player.get_min_state(new_state)[0]
        [_, ref, rots] = player.get_min_state(new_state)[1]
        i += 1
        if old_key:
            if player.role == 'X':
                reward = -reward

            playerX.get_step_info(old_key, old_action, reward, new_key)
            playerO.get_step_info(old_key, old_action, reward, new_key)

        old_key = key
        key = new_key
        old_action = action

        if i % 2 == 0:
            player = playerX
        else:
            player = playerO

        if done:
            reward = -reward
            playerX.get_step_info(old_key, old_action, reward, None)
            playerO.get_step_info(old_key, old_action, reward, None)
            break

    if show:
        board.show_board()

    winner_value = board.is_game_over()
    board.reset()
    return winner_value


def play_n_duels(games, agent1, agent2, show=False):
    playerX = agent1
    playerO = agent2
    frec = []
    draws = 0
    for k in range(1, games+1):
        winner_value = duel(playerX, playerO, show)
        if winner_value == 2:
            playerO.wins += 1
            aux = playerX
            playerX = playerO
            playerO = aux
            playerX.role = 'X'
            playerO.role = 'O'
        elif winner_value == 1:
            playerX.wins += 1 
        elif winner_value == 0:
            draws += 1 

        if k % 100 == 0:
            frec.append((agent2.wins+draws)/k)

    return np.array(frec)
