from board import Board

board = Board()
PLAYERS = ['X', 'O']
def duel(agent_X, agent_O, show=False):
    board.reset()
    done = False
    k = 0
    key = 0
    reflected = False
    rots = 0
    while not done:
        player = PLAYERS[k % 2]
        if k % 2 == 0:
            agent = agent_X
        elif k % 2 != 0: 
            agent = agent_O
        if show:
            print(reflected,rots)
            board.show_board()
        action = agent.select_action(key)
        board_action = agent.get_board_action(action, reflected, rots)
        state, reward, done = board.step(board_action, player)
        key = agent.get_min_state(state)[0]
        [_, reflected, rots] = agent.get_min_state(state)[1]
        k +=1
    board.show_board()
    return board.is_game_over()

