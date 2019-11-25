#!/usr/bin/env python3
from .base_agent import BaseAgent
from ast import literal_eval
from collections import Counter
import operator
import json
import numpy as np
from .read_tables import REWARDS


class MinMax_Agent(BaseAgent):
    '''
    Define el jugador que sigue el algritmo de min max
    '''

    def __init__(self):
        '''
        Constructor
        '''
        BaseAgent.__init__(self)
        self.rewardsX = self.minmax()
        self.rewardsO = self.minmax(X=False)

    def select_action(self, state):
        if self.role == 'X':
            rewards = self.rewardsX
        else:
            rewards = self.rewardsO
            
        transitions = {k: v for k, v in rewards.items() if k[0] == state}
        _, a, _ = max(transitions.items(), key=operator.itemgetter(1))[0]
        return a

    def get_unique_states(self, transitions, position):
        return set(map(lambda x: x[position], transitions))
   
    def get_rewards_for_O(self):
        return {k: -v for k, v in REWARDS.items() if k[0] != 0}

    def minmax(self, X=True):
        rewards = REWARDS
        if not X:
            rewards = self.get_rewards_for_O()

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
                        if X:
                            if turn == 'X':
                                f = min
                            else:
                                f = max
                        else:
                            if turn == 'X':
                                f = max
                            else:
                                f = min
                        rewards[(sm, am, sm_plus_1)] = f(rm, rn)
                        aux = [s] + keys + visited
                        if sn not in aux:
                            keys.append(sn)

            visited.append(s)
        return rewards
