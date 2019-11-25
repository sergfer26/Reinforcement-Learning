import json
import os
from collections import defaultdict
from ast import literal_eval
import collections


def remap_keys(mapping, type_=float):
    dic = defaultdict(type_)
    for k, v in mapping.items():
        dic[str(k)] = v
    return dic


def remap_transits(mapping, type_=collections.Counter):
    dic = defaultdict(type_)
    for key, val in mapping.items():
        for k, v in val.items():
            dic[str(key)][str(k)] = v
    return dic


def remap_stringkeys(mapping, type_=float):
    dic = defaultdict(type_)
    for k, v in mapping.items():
        k = literal_eval(k)
        dic[k] = v
    return dic


def remap_values(mapping, type_=float):
    dic = defaultdict(type_)
    for k, v in mapping.items():
        dic[k] = -v
    return dic


# tvalues_fn = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'tables/trained_values.txt')
# TVALUES = json.load(open(tvalues_fn))
# TVALUES = remap_stringkeys(TVALUES)
# 
rewards_fn = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'tables/minmax_rewards.txt')
rewards = json.load(open(rewards_fn))
REWARDS = remap_stringkeys(rewards)
# 
# qvalues_fn = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'tables/qvalues.txt')
# qvalues = json.load(open(qvalues_fn))
# QVALUES = remap_stringkeys(qvalues)
# 
# values_fn = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'tables/values.txt')
# rewards = json.load(open(values_fn))
# VALUES = remap_stringkeys(rewards)
