#!/usr/bin/env python3
from board import Board as ttt
from agent_value_iteration import AgentVI as avi
from base_agent import remap_keys
from collections import Counter
import numpy as np
import json
from numpy.linalg import norm as norma

if __name__ == "__main__":
    player = avi()
    matrix = []
    player.play_n_random_games(2000)
    dim = len(player.values)
    y = np.repeat(0, dim)
    epsilon = 0.01
    i = 0
    while True:
        player.value_iteration()
        x = np.array(list(player.values.values()))
        matrix.append(x)
        i += 1
        if norma(x-y) < epsilon:
            break
        y = x
    matrix = np.flip(matrix, axis=0)
    matrix.reshape((i, dim))

    rewards = remap_keys(player.rewards)
    json.dump(player.values, open("values.txt",'w'))
    json.dump(rewards, open("rewards.txt",'w'))

    # d2 = json.load(open("text.txt"))
    





    