from agent_tabular_qlearning import Agent_TQL as atql
import base_agent as ba
import numpy as np 
from numpy.linalg import norm as norma
import json

if __name__ == '__main__':
    player = atql()
    ref = False
    rots = 0
    players = ['X', 'O']
    player.values = ba.remap_stringkeys_rewards(json.load(open("qvalues.txt")))
    dim = len(player.values)
    y = np.repeat(0, dim)
    epsilon = 0.001
    i = 0
    while True:
        k, a, r, nk, ref, rots, done = player.sample_env(ref, rots, players[i % 2])
        player.value_update(k, a, r, nk)
        if done: 
            print('final')
            print(player.values[(k, a)])
            player.key = 0
            k = 0
            nk = 0
            player.board.reset()
        x = np.array(list(player.values.values()))
        i +=1
        if i > 10000 and norma(x-y) < epsilon:
            break
        y = x
    print(i)

    values = ba.remap_keys(player.values)
    json.dump(values, open("qvalues_trained.txt",'w'))
