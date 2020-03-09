#!/usr/bin/env python3
from torch.utils.tensorboard import SummaryWriter
from .board import Board as board
from .human import Human as human
from .base_agent import BaseAgent
from random import shuffle 
import json
import numpy as np
import pandas as pd

import pdb; pdb.set_trace()


def get_turn(state):
    role = BaseAgent.check_turn(state)
    if role == 'X':
        return 0
    else:
        return 1


def duel(playerX, playerO, show=False, board=board(), old_key=None, old_action=None):
    '''
    Realiza una partida entre dos jugadores
    '''
    playerX.role = 'X'
    playerO.role = 'O'
    done = False
    state = board.state
    key = BaseAgent.get_min_state(state)[0]
    [_, ref, rots] = BaseAgent.get_min_state(state)[1]
    i = get_turn(key)
    if i == 0:
        player = playerX
    else:
        player = playerO
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

    import pdb; pdb.set_trace()
def play_n_duels(games, agent1, agent2, show=False):
    '''
    Realiza n partidas entre dos jugadores y mide el rendimiento 
    del segundo.
    '''
    writer = SummaryWriter()
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
        elif winner_value == 1:
            playerX.wins += 1 
        elif winner_value == 0:
            draws += 1 

        if k % 100 == 0:
            writer.add_scalar('Juegos no perdidos', (agent2.wins+draws)/k, k)
            frec.append((agent2.wins+draws)/k)
        
    return np.array(frec)


def championship(players):

    shuffle(players)
    winners = []
    if len(players) == 1:
        import pdb; pdb.set_trace()
        return players

    while players:
        playerX = players.pop()
        playerO = players.pop()
        winner = duel(playerX, playerO)
        if winner in set([0, 1]):
            playerX.wins += 1
            winners.append(playerX)
        else:
            playerO.wins += 1
            winners.append(playerO)

    championship(winners)

        

