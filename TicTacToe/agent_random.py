#!/usr/bin/env python3
from .base_agent import BaseAgent
from numpy.random import choice


class Random(BaseAgent):

    def select_action(self, key):
        state = self.key_to_state(key)
        actions = [i for i, e in enumerate(state) if e == 0]
        action = choice(actions)
        return action
