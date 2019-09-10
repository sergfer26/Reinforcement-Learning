from agent import Agent


class Agent_value_iteration(Agent):
    '''
        Agente basado en el método de value iteration
    '''

    def calc_action_value(self, state, action, index):
        key = self.state_to_base10(state)
        target_counts = self.transits[(key, action, index)].items()
        tgt_key, count = list(target_counts)[0]
        reward = self.rewards[(key, action, tgt_key)]
        action_value = count*(reward + self.gamma * self.values[tgt_key])
        # for tgt_key, count in target_counts.items():
        #    reward = self.rewards[(key, action, tgt_key)]
        #    action_value += (count/total)*(reward + self.gamma * self.values[tgt_key])
        return action_value

    def select_action(self, state):
        best_action, best_value, = None, None
        index = self.get_state_index(state)
        key = self.state_to_base10(state)
        for action in range(self.board.action_space.n):

            if (key, action, index) in self.transits.keys():
                action_value = self.calc_action_value(state, action, index)
                if best_value is None or best_value < action_value:
                    best_value = action_value
                    best_action = action
            else:
                pass
            return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        k = 0 
        players = ['X', 'O']
        while True:
            action = self.select_action(state)
            key = self.state_to_base10(state)
            new_state, reward, is_done = env.step(action, players[k % 2])
            new_key = self.state_to_base10(new_state)
            self.rewards[(key, action, new_key)] = reward
            index = self.get_state_index(state)
            self.transits[(key, action, index)][new_key] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
            k += 1 
        return total_reward

    def check_mark(self, action, state):
        if state[action] == 0:
            return True
        return False

    def value_iteration(self):
        for key in self.values.keys(): 
            state_values = []
            state = self.base10_to_state(key)
            index = self.get_state_index(state)
            for action in range(self.board.action_space.n):
                if (key, action, index) in self.transits.keys():
                    state_values.append(self.calc_action_value(state, action, index))  
            self.values[key] = [max(state_values) if len(state_values) else 0]