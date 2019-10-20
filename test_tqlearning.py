from agent_tabular_qlearning import Agent_TQL as atql
from base_agent import remap_keys
import numpy as np 
from numpy.linalg import norm as norma
import json

if __name__ == '__main__':
    player = atql()
    ref = False
    rots = 0
    players = ['X', 'O']
    matrix = []
    player.play_n_random_games(2000)
    dim = len(player.values)
    y = np.repeat(0, dim)
    epsilon = 0.01
    for i in range(10000):
        k, a, r, nk, ref, rots, done = player.sample_env(ref, rots, players[i % 2])
        player.value_update(k, a, r, nk)
        if done: 
            player.key = 0
            player.board.reset()
        x = np.array(list(player.values.values()))
        matrix.append(x)
        i += 1
        #if norma(x-y) < epsilon:
        #    break
        y = x
    matrix = np.flip(matrix, axis=0)
    matrix.reshape((i, dim))

    values = remap_keys(player.values)
    json.dump(values, open("qvalues.txt",'w'))
