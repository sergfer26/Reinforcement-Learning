from agent import Agent


class Agent_value_iteration(Agent):
    '''
        Agente basado en el método de value iteration
    '''

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count/total)*(reward + self.gamma * self.values[tgt_state])
        return action_value

    def select_action(self, state):
        best_action, best_value, = None, None
        for action in range(self.board.action_space.n):
            if (state, action) in self.transits.keys():
                action_value = self.calc_action_value(state, action)
                if best_value is None or best_value < action_value:
                    best_value = action_value
                    best_action = action
            else:
                pass
            return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def check_mark(self, action):
        if self.board.state[action] == 0:
            return True
        return False

    def value_iteration(self):
        state_values = []
        for key in self.keys: 
            state = self.base10_to_state(key)
            for action in range(self.board.action_space.n):
                bool = self.check_mark(action)
                if bool: 
                    state_values.append(self.calc_action_value(state, action))
            self.values[state] = max(state_values)