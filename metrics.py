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
from scipy.spatial.distance import euclidean

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

def actual_offensive_impact(actions, player_id, game_id):
    offensive_actions = ['pass', 'cross', 'dribble', 'take-on', 'shot']
    player_actions = actions[(actions['player_id'] == player_id) &
                             (actions['game_id'] == game_id) &
                             (actions['type_name'].isin(offensive_actions))]
    return player_actions['vaep_value'].sum()

def expected_offensive_impact(actions, player_id, current_game_id, minutes_df):
    current_game = actions[actions['game_id'] == current_game_id]['game_id'].iloc[0]
    past_games = actions[(actions['player_id'] == player_id) &
                         (actions['game_id'] < current_game)]

    total_minutes = minutes_df[(minutes_df['player_id'] == player_id) &
                               (minutes_df['game_id'] < current_game)]['minutes_played'].sum()

    if total_minutes == 0:
        return 0.0  # jogador não jogou antes

    oi_total = 0
    for gid in past_games['game_id'].unique():
        oi_total += actual_offensive_impact(actions, player_id, gid)
    return (oi_total * 90) / total_minutes

def responsibility_share(player1_pos, player2_pos, opponent_pos):
    position_map = {
        'CB': (2, 1), 'RB': (4, 1), 'LB': (0, 1), 'DM': (2, 2),
        'CM': (2, 3), 'RW': (4, 4), 'LW': (0, 4), 'ST': (2, 4),
    }
    dist1 = euclidean(position_map[player1_pos], position_map[opponent_pos])
    dist2 = euclidean(position_map[player2_pos], position_map[opponent_pos])
    return (1/(dist1+1e-5) + 1/(dist2+1e-5)) / 2

def joint_defensive_impact(actions, minutes_df, player1_id, player2_id, game_id, player_positions):
    opponents = actions[(actions['gameId'] == game_id)]['playerId'].unique()
    jdi = 0

    for opponent_id in opponents:
        actual_oi = actual_offensive_impact(actions, opponent_id, game_id)
        expected_oi = expected_offensive_impact(actions, opponent_id, game_id, minutes_df)
        diff = expected_oi - actual_oi
        
        pos1 = player_positions.get(player1_id, 'CB')
        pos2 = player_positions.get(player2_id, 'CB')
        opponent_pos = player_positions.get(opponent_id, 'ST')

        resp = responsibility_share(pos1, pos2, opponent_pos)

        shared_minutes = minutes_df[
            (minutes_df['player_id'].isin([player1_id, player2_id, opponent_id])) &
            (minutes_df['game_id'] == game_id)
        ]['minutes_played'].min()  # minutos em comum

        jdi += diff * resp * (90 / shared_minutes if shared_minutes else 0)

    return jdi

def calculate_jdi90(actions, minutes_df, player1_id, player2_id, game_ids, player_positions):
    total_jdi = 0
    total_minutes = 0

    for gid in game_ids:
        jdi_match = joint_defensive_impact(actions, minutes_df, player1_id, player2_id, gid, player_positions)
        minutes = minutes_df[
            (minutes_df['player_id'].isin([player1_id, player2_id])) &
            (minutes_df['game_id'] == gid)
        ]['minutes_played'].min()
        if minutes:
            total_jdi += jdi_match
            total_minutes += minutes

    return (total_jdi * 90) / total_minutes if total_minutes else 0
