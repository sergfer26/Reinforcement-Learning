from agent import Agent
import collections

class Agent_TQL(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.values = collections.defaultdict(float) 
        self.alpha = 0.2

    def sample_env(self, player, reflected, rots):
        state = self.base10_to_state(self.key)
        action = self.select_random_action(state)
        board_action = self.get_board_action(action, reflected, rots)
        old_key = self.key
        new_state, reward, is_done = self.board.step(board_action, player)
        new_key = self.get_min_state(new_state)[0]
        [_, reflected, rots] = self.get_min_state(new_state)[1]
        self.values[(self.key, action)] = 0
        self.key = self.reset_key if is_done else new_key
        return old_key, action, reward, new_key, reflected, rots

    def best_value_and_action(self, key):
        best_value, best_action = None, None
        values = list(self.values.keys())
        actions = [ a for k, a in values if key == k]
        import pdb; pdb.set_trace()
        for action in actions:
            action_value = self.values[(key, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, k, a, r, nk):
        best_value, _ = self.best_value_and_action(nk)
        new_val = r + self.gamma * best_value
        old_val = self.values[(k, a)]
        self.values[(k, a)] = old_val * (1- self.alpha) + new_val * self.alpha

    def play_episode(self, board):
        total_reward = 0.0
        key = 0
        reflected = False
        rots = 0
        self.board.reset()
        players = ['X', 'O']
        k = 0
        while True:
            _, action = self.best_value_and_action(key)
            board_action = self.get_board_action(action, reflected, rots)
            new_state, reward, is_done = board.step(board_action, players[k % 2])
            new_key = self.get_min_state(new_state)[0]
            [_, reflected, rots] = self.get_min_state(new_state)[1]
            total_reward += reward
            if is_done:
                break
            key = new_key
            k += 1
        return total_reward
