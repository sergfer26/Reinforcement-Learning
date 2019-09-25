from board import TicTacToe as ttt
from agent_tabular_qlearning import Agent_TQL as atql
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    board = ttt()
    player = atql()
    TE = 200
    reflected = False
    rots = 0
    i = 0
    best_reward = 0.0
    players = ['X', 'O']
    while True:
        k, a, r, nk, reflected, rots = player.sample_env(players[i % 2], reflected, rots)
        player.value_update(k, a, r, nk)

        reward = 0.0
        for _ in range(TE):
            reward += player.play_episode(board)
        reward /= TE
        i += 1
        writer.add_scalar('reward', reward, i)
        if reward > best_reward:
            print('Best reward updated %.3f -> %.3f' % (best_reward, reward))
            best = reward
        if reward > 0.70:
            print('Solved in %d iterations!' % i)
            break
    writer.close()
