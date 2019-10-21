import matplotlib.pyplot as plt
from duel import duel
from agent_Value_iter import Value_iter
from agent_random import Random
from human import Human
def rendimiento(player_X,player_O,K):
    historia = [0]
    for i in range(1,K+1):
        x = duel(player_X,player_O,show = False)
        if x == 1:
            historia.append(historia[-1] + 1)
    historia = historia[1:]
    for i in range(0,len(historia)):
        historia[i] =historia[i]/(i+1)
    return historia 

playerX = Value_iter()
playerO = Random()

print(rendimiento(playerX,playerO,10))
        
