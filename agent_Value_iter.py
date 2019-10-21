#!/usr/bin/env python3
from duel import duel
from board import Board
from agent import AgentVI
import json
from agent_random import Random 
from base_agent import remap_stringkeys_rewards,remap_stringkeys_transits 
def Value_iter():
    player_X = AgentVI()
    player_X.transits = remap_stringkeys_transits(json.load(open("transits.txt")))
    player_X.rewards =  remap_stringkeys_rewards(json.load(open("rewards.txt")))
    player_X.values =  remap_stringkeys_rewards(json.load(open("values.txt")))

    return player_X
