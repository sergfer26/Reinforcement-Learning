#!/usr/bin/env python3
from .base_agent import BaseAgent


class Human(BaseAgent):

    def select_action(self, key, ref, rots):
        state = self.key_to_state(key)
        for _ in range(rots):
            state = self.rotate_left(state)
        if ref:
            if rots % 2 == 0:
                state = self.reflect_h(state)
            else: 
                state = self.reflect_v(state)
        actions = [i for i, e in enumerate(state) if e == 0]
        # print(actions)
        action = int(input('Tu jugada es: '))
        return action

        
        
