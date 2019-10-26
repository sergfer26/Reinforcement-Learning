#!/usr/bin/env python3
from duel import duel
from board import Board
from agent_random import Random
from agent_value_iteration import AgentVI as avi
from agent_value_iteration import create_avi
from human import Human

Agent1 = create_avi()
Agent2 = Random()
Agent3 = Human()
A1, A2, A3 = 0, 0, 0
Agents = [Agent1, Agent2, Agent3]
names = ['Value_iteration', 'Aleatorio', 'Humano']


def un_duelo(X, O, varX, varO):
    gana = 1 if duel(X, O, show=True) == 1 else 0
    if gana == 1:
        varX += 1
    else:
        varO += 1
    return varX, varO


A1, A2 = un_duelo(Agent1, Agent2, A1, A2)
A1, A3 = un_duelo(Agent1, Agent3, A1, A3)

A2, A1 = un_duelo(Agent2, Agent1, A2, A1)
A2, A3 = un_duelo(Agent2, Agent3, A2, A3)

A3, A2 = un_duelo(Agent2, Agent2, A3, A2)
A3, A1 = un_duelo(Agent3, Agent1, A3, A1)


print('Puntuacion final:')
print(names[0] + ' ' + str(A1))
print(names[1] + ' ' + str(A2))
print(names[2] + ' ' + str(A3))
