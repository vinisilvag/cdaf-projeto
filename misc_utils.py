import pandas as pd
import socceraction.spadl as spd
from tqdm import tqdm
import random
import numpy as np
import xgboost as xgb
import sklearn.metrics as mt

def spadl_transform(events, matches):
    spadl = []
    game_ids = events.game_id.unique().tolist()
    for g in tqdm(game_ids):
        match_events = events.loc[events.game_id == g]
        match_home_id = matches.loc[(matches.matchId == g) & (matches.side == 'home'), 'teamId'].values[0]
        match_actions = spd.wyscout.convert_to_actions(events=match_events, home_team_id=match_home_id)
        match_actions = spd.play_left_to_right(actions=match_actions, home_team_id=match_home_id)
        match_actions = spd.add_names(match_actions)
        spadl.append(match_actions)
    spadl = pd.concat(spadl).reset_index(drop=True)
    return spadl

def select_random_games(spadl_dict):
    """
    Retorna { liga: game_id } escolhendo aleatoriamente
    uma partida de cada liga.
    """
    return {
        league: random.choice(df['game_id'].unique().tolist())
        for league, df in spadl_dict.items()
    }
    
def features_transform(spadl):
    spadl.loc[spadl.result_id.isin([2, 3]), ['result_id']] = 0
    spadl.loc[spadl.result_name.isin(['offside', 'owngoal']), ['result_name']] = 'fail'

    xfns = [
        ft.actiontype_onehot,
        ft.bodypart_onehot,
        ft.result_onehot,
        ft.goalscore,
        ft.startlocation,
        ft.endlocation,
        ft.team,
        ft.time,
        ft.time_delta
    ]

    features = []
    for game in tqdm(np.unique(spadl.game_id).tolist()):
        match_actions = spadl.loc[spadl.game_id == game].reset_index(drop=True)
        match_states = ft.gamestates(actions=match_actions)
        match_feats = pd.concat([fn(match_states) for fn in xfns], axis=1)
        features.append(match_feats)
    features = pd.concat(features).reset_index(drop=True)

    return features

def labels_transform(spadl):
    yfns = [lab.scores, lab.concedes]

    labels = []
    for game in tqdm(np.unique(spadl.game_id).tolist()):
        match_actions = spadl.loc[spadl.game_id == game].reset_index(drop=True)
        labels.append(pd.concat([fn(actions=match_actions) for fn in yfns], axis=1))

    labels = pd.concat(labels).reset_index(drop=True)

    return labels

def train_vaep(X_train, y_train, X_test, y_test):
    models = {}
    for m in ['scores', 'concedes']:
        models[m] = xgb.XGBClassifier(random_state=0, n_estimators=50, max_depth=3)

        print('training ' + m + ' model')
        models[m].fit(X_train, y_train[m])

        p = sum(y_train[m]) / len(y_train[m])
        base = [p] * len(y_train[m])
        y_train_pred = models[m].predict_proba(X_train)[:, 1]
        train_brier = mt.brier_score_loss(y_train[m], y_train_pred) / mt.brier_score_loss(y_train[m], base)
        print(m + ' Train NBS: ' + str(train_brier))
        print()

        p = sum(y_test[m]) / len(y_test[m])
        base = [p] * len(y_test[m])
        y_test_pred = models[m].predict_proba(X_test)[:, 1]
        test_brier = mt.brier_score_loss(y_test[m], y_test_pred) / mt.brier_score_loss(y_test[m], base)
        print(m + ' Test NBS: ' + str(test_brier))
        print()

        print('----------------------------------------')

    return models

def generate_predictions(features, models):
    preds = {}
    for m in ['scores', 'concedes']:
        preds[m] = models[m].predict_proba(features)[:, 1]
    preds = pd.DataFrame(preds)
    return preds

def calculate_action_values(spadl, predictions):
    action_values = fm.value(actions=spadl, Pscores=predictions['scores'], Pconcedes=predictions['concedes'])
    action_values = pd.concat([
        spadl[['original_event_id', 'action_id', 'game_id', 'start_x', 'start_y', 'end_x', 'end_y', 'type_name', 'result_name', 'player_id',]],
        predictions.rename(columns={'scores': 'Pscores', 'concedes': 'Pconcedes'}),
        action_values
    ], axis=1)
    return action_values