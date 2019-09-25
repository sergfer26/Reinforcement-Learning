#!/usr/bin/env python3
from board import TicTacToe as ttt
from agent_value_iteration import Agent_value_iteration as agent_vi
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    TE = 100  # test episodes
    test_board = ttt()
    player = agent_vi()
    writer = SummaryWriter(comment='v-learning')
    i = 0
    best_reward = 0.0
    player.play_n_random_steps(6000)
    while True:
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

    print(player.values)

