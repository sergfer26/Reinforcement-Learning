#!/usr/bin/env python3
from .base_agent import BaseAgent
from numpy.random import choice


class RandomAgent(BaseAgent):

    def __init__(self):
        BaseAgent.__init__(self)

    def select_action(self, key):
        return self.select_random_action(key)
