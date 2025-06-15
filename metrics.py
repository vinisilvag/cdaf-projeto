from socceraction.vaep import features as ft
from socceraction.vaep import labels as lab
from socceraction.vaep import formula as fm
import pandas as pd
import socceraction.spadl as spd
from tqdm import tqdm
import random
import numpy as np
import xgboost as xgb
import sklearn.metrics as mt

def extended_vaep(interaction):
    current_action, next_action = interaction
    return current_action["vaep_value"] + next_action["vaep_value"]

def get_interactions(actions, game_id, player_before, player_after):
    desired_actions = ['pass', 'cross', 'dribble', 'take-on', 'shot']
    
    game_actions = actions[actions['game_id'] == game_id]
    filtered = game_actions[game_actions['type_name'].isin(desired_actions)]
    sorted_data = filtered.sort_values(by=['period_id', 'time_seconds']).reset_index(drop=True)
    
    interactions = []
    
    for i in range(len(sorted_data) - 1):
        current_action = sorted_data.iloc[i]
        next_action = sorted_data.iloc[i + 1]
        if (current_action["player_id"] == player_before) and (next_action["player_id"] == player_after):
            interactions.append((current_action, next_action))        
    
    return interations

def joint_offensive_impact(actions, game_id, p, q):
    interactions = get_interactions(actions, game_id, p, q)
    interactions_reverse = get_interactions(actions, game_id, q, p)
    interactions_sum = 0
    interactions_reverse_sum = 0

    for i in interactions:
        interactions_sum += extended_vaep(i)

    for i in interactions_reverse:
        interactions_reverse_sum += extended_vaep(i)
    
    return interactions_sum + interactions_reverse_sum
    
def offensive_impact():
    pass

def expected_offensive_impact():
    pass

def responsability():
    pass

def joint_defensive_impact():
    pass
