#!/usr/bin/env python3
from base_agent import BaseAgent   
from ast import literal_eval
from collections import Counter
import operator
import json
import numpy as np

REWARDS = json.load(open('rewards.txt'))


class MinMax_Agent(BaseAgent):
    '''
    Define el jugador que sigue el algritmo de min max
    '''

    def __init__(self):
        '''
        Constructor
        '''
        BaseAgent.__init__(self)
        self.rewards = {literal_eval(k): v for k, v in REWARDS.items()}

    def select_action(self, state):
        transitions = {k: v for k, v in self.rewards.items() if k[0] == state}
        k, a, nk = max(transitions.items(), key=operator.itemgetter(1))[0]
        print(self.rewards[(k, a, nk)])
        return a

    def get_unique_states(self, transitions, position):
        return set(map(lambda x: x[position], transitions))

    def check_turn(self, state):
        state_representation = self.key_to_state(state)
        c = Counter(state_representation)
        if c[1] == c[2]:
            return 'X'
        else:
            return 'O'

    def minmax(self):
        rewards = self.rewards
        keys = list(self.get_unique_states(rewards.keys(), 2) - 
                    self.get_unique_states(rewards.keys(), 0))
        visited = []
        while keys:
            s = keys.pop(0)

            sns = [(sn, rn) for (sn, an, sn_plus_1), rn in rewards.items() if s == sn_plus_1]
            assert all(_r == sns[0][1] for _s, _r in sns)

            for sn, rn in sns:
                for sm, am, sm_plus_1 in rewards.keys():
                    if sm_plus_1 == sn:
                        rm = rewards[(sm, am, sm_plus_1)]
                        turn = self.check_turn(sm)
                        if turn == 'X':
                            f = min
                        else:
                            f = max
                        rewards[(sm, am, sm_plus_1)] = f(rm, rn)
                        aux = [s] + keys + visited
                        if sn not in aux:
                            keys.append(sn)

            visited.append(s)
        self.rewards = rewards


