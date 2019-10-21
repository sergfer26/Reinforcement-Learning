#!/usr/bin/env python3
from base_agent import BaseAgent

class Human(BaseAgent):
    def select_action(self,key):
        state = self.key_to_state(key)
        actions = [i for i, e in enumerate(state) if e == 0]
        print(actions)
        action = int(input('Tu jugada es: '))
        return action

        
        
