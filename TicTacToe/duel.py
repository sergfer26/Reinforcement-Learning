from board import Board
from human import Human as human
import json

board = Board()
PLAYERS = ['X', 'O']
rewards = json.load(open('rewards.txt'))


def duel(agent_X, agent_O, show=False):
    board.reset()
    done = False
    k = 0
    key = 0
    ref = False
    rots = 0
    while not done:
        player = PLAYERS[k % 2]

        if k % 2 == 0:
            agent = agent_X
        elif k % 2 != 0:
            agent = agent_O

        if show:
            board.show_board()
        # import pdb; pdb.set_trace()
        if isinstance(agent, human):
            action = agent.select_action(key, ref, rots)
            board_action = action
        else:
            action = agent.select_action(key)
            board_action = agent.get_board_action(action, ref, rots)

        new_state, reward, done = board.step(board_action, player)
        new_key = agent.get_min_state(new_state)[0]
        [_, ref, rots] = agent.get_min_state(new_state)[1]
        k += 1
        agent_X.get_step_info(key, action, reward, new_key)
        agent_O.get_step_info(key, action, -reward, new_key)
        key = new_key

    board.show_board()
    return board.is_game_over()

    # def do_n_duels(n, agent_X, agent_O):
