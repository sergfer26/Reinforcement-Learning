#!/usr/bin/env python3
from board import TicTacToe as ttt
from agent_value_iteration import Agent_value_iteration as avi
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    TE = 100  # test episodes
    test_board = ttt()
    player = avi()
    writer = SummaryWriter(comment='v-learning')
    i = 0
    best_reward = 0.0
    while True:
        player.play_n_random_games(400)
        i += 1
        player.value_iteration()
        reward = 0.0 
        for _ in range(TE):
            reward += player.play_episode(test_board)
        reward /= TE
        writer.add_scalar('reward', reward, i)
        if reward > best_reward:
            print("Best reward updated % .3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in  %d iteration!" % i)
            break 
    writer.close()


import numpy as np
from agent_value_iteration import Agent_value_iteration as avi
from test_heatmap import plot_heatmap

player = avi()
player.play_n_random_games(4000)
n = len(player.values_to_list())
k = 0
z = np.array([])
y = np.zeros((1, n), dtype=float)
while True:
    player.value_iteration()
    x = np.array(player.values_to_list())
    if np.linalg.norm(x-y) == 0:
        z = np.vstack([z, x])
        break 
    else: 
        y = x
print(z)




    