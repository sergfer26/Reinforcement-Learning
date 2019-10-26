#!/usr/bin/env python3
from duel import duel
from board import Board
from agent_value_iteration import AgentVI as avi
import json
from human import Human
from minmax_agent import MinMax_Agent as mma
from base_agent import remap_stringkeys
player_X = avi()

player_X.rewards = remap_stringkeys(json.load(open("rewards.txt")))
player_X.values = remap_stringkeys(json.load(open("values.txt")))

player_O = Human()

print(duel(player_X, player_O, show=True))
