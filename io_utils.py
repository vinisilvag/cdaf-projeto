import pandas as pd
from webscraping import get_best_match

def save_players_merged(path):
    players = pd.read_json(path_or_buf=path + "players.json")
    players_scrapped = pd.read_csv(path + "players_scrapped.csv")
    players['player_name'] = players['firstName'] + ' ' + players['lastName']
    players['player_name'] = players['player_name'].apply(lambda x: x.encode('utf-8').decode('unicode_escape'))

    player_names = players_scrapped["Player"].tolist()

    players["matched_name"] = players["player_name"].apply(lambda x: get_best_match(x, player_names))

    players.dropna(subset=["matched_name"], inplace=True)
    
    merged_players = pd.merge(players, players_scrapped, left_on="matched_name", right_on="Player", how="inner")

    merged_players = merged_players.rename(columns={'wyId': 'player_id'})
    
    return merged_players
    
def load_matches(path):
    matches = pd.read_json(path_or_buf=path)
    # as informações dos times de cada partida estão em um dicionário dentro da coluna 'teamsData', então vamos separar essas informações
    team_matches = []
    for i in range(len(matches)):
        match = pd.DataFrame(matches.loc[i, 'teamsData']).T
        match['matchId'] = matches.loc[i, 'wyId']
        team_matches.append(match)
    team_matches = pd.concat(team_matches).reset_index(drop=True)
    return team_matches

def load_players(path):
    players = pd.read_csv(path + "players_merged.csv")
    return players

def load_events(path):
    events = pd.read_json(path_or_buf=path)
    # pré processamento em colunas da tabela de eventos para facilitar a conversão p/ SPADL
    events = events.rename(columns={
        'id': 'event_id',
        'eventId': 'type_id',
        'subEventId': 'subtype_id',
        'teamId': 'team_id',
        'playerId': 'player_id',
        'matchId': 'game_id'
    })
    events['milliseconds'] = events['eventSec'] * 1000
    events['period_id'] = events['matchPeriod'].replace({'1H': 1, '2H': 2})
    return events

def load_minutes_played_per_game(path):
    minutes = pd.read_json(path_or_buf=path)
    minutes = minutes.rename(columns={
        'playerId': 'player_id',
        'matchId': 'game_id',
        'teamId': 'team_id',
        'minutesPlayed': 'minutes_played'
    })
    minutes = minutes.drop(['shortName', 'teamName', 'red_card'], axis=1)
    return minutes