from agent import Agent


class Agent_value_iteration(Agent):
    '''
        Agente basado en el método de value iteration
    '''

    def calc_action_value(self, key, action):
        transits = self.transits[(key, action)]
        tgt_key = sum(transits.keys())
        count = sum(transits.values())
        reward = self.rewards[(key, action, tgt_key)]
        action_value = reward + self.gamma * self.values[tgt_key]
        return action_value

    def select_action(self, key):
        best_action, best_value, = None, None
        transits = list(self.transits.keys())
        actions = [a for k, a in transits if key == k]
        for action in actions:
            action_value = self.calc_action_value(key, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
            else:
                pass
            return best_action

    def play_episode(self, env):
        total_reward = 0.0
        env.reset()
        key = 0
        k = 0 
        reflected = False
        rots = 0
        players = ['X', 'O']
        while True:
            action = self.select_action(key)
            board_action = self.get_board_action(action, reflected, rots)
            new_state, reward, is_done = env.step(board_action, players[k % 2])
            new_key = self.get_min_state(new_state)[0]
            [_, reflected, rots] = self.get_min_state(new_state)[1]
            self.rewards[(key, action, new_key)] = reward
            self.transits[(key, action)][new_key] += 1
            total_reward += reward
            if is_done:
                break
            key = new_key
            k += 1 
        return total_reward

    def check_mark(self, action, state):
        if state[action] == 0:
            return True
        return False

    def value_iteration(self):
        transits = list(self.transits.keys())
        for key in list(self.values.keys()): 
            state_vals = []
            val = self.values[key]
            actions = [a for k, a in transits if key == k]
            for action in actions:
                state_vals.append(self.calc_action_value(key, action))  
            self.values[key] = max(state_vals) if len(state_vals) else val
    
    def get_sorted_values(self):
        turns = self.turns
        vals = self.values
        sorted_items = sorted(vals.items(), key=lambda x: turns.get(x[0])) 
        return dict(sorted_items)
        
    def values_to_list(self):
        values = self.get_sorted_values().values()
        return list(values)

    

