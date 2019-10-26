#!/usr/bin/env python3
from duel import duel
from board import Board
from agent import AgentVI
import json
from minmax_agent import MinMax_Agent as mma
from base_agent import remap_stringkeys_rewards, remap_stringkeys_transits

player_X = AgentVI()

player_X.transits = remap_stringkeys_transits(json.load(open("transits.txt")))
player_X.rewards = remap_stringkeys_rewards(json.load(open("rewards.txt")))
player_X.values = remap_stringkeys_rewards(json.load(open("values.txt")))


player_O = mma()

print(duel(player_X, player_O, show=True))
