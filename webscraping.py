import pandas as pd
from collections import defaultdict
import requests
from bs4 import BeautifulSoup, Comment
from thefuzz import fuzz, process

def get_data_by_league(league_data):
    scrapped_data = []
    for index, (url, table_id) in enumerate(league_data):
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        
        # Faz requisição
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # A tabela está dentro de um comentário HTML
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        
        # Busca pela tabela dentro dos comentários
        table = None
        for comment in comments:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', id=table_id)
            if table:
                break
        
        # Se tabela foi encontrada, extrair os dados
        if not table:
            raise ValueError(f'Table {table_id} not found!')
            
        # Extrair cabeçalhos
        thead = table.find('thead')
        headers = []
        for tr in thead.find_all('tr'):
            if 'over_header' in tr.get('class', []):
                continue
            headers.extend([th.text.strip() for th in tr.find_all('th')])

        headers = headers[1:len(headers) - 1]

        if table_id == "stats_passing":
            labels = ['_Total', '_Short', '_Medium', '_Long']
            counts = defaultdict(int)
            new_stats_passing = []
            for col in headers:
                if col in ['Cmp', 'Att', 'Cmp%']:
                    suffix = labels[counts[col]]
                    if col == "Att" or col == "Cmp":
                        new_stats_passing.append(col + suffix + "_Passing")
                    else:
                        new_stats_passing.append(col + suffix)
                    counts[col] += 1
                else:
                    new_stats_passing.append(col)
            headers = new_stats_passing

        if table_id == "stats_defense":
            labels = ['_Tackles', '_Challenges']
            counts = defaultdict(int)
            new_stats_defense = []
            for col in headers:
                if col in ['Tkl']:
                    suffix = labels[counts[col]]
                    new_stats_defense.append(col + suffix)
                    counts[col] += 1
                if col in ["Att", "Blocks", "Int", "Lost", "TklW"]:
                    new_stats_defense.append(col + "_Defense")
                else:
                    new_stats_defense.append(col)
            headers = new_stats_defense

        if table_id == "stats_passing_types":
            new_stats_passing_types = []
            for col in headers:
                if col in ["Att", "Blocks", "Cmp", "Crs", "Off"]:
                    new_stats_passing_types.append(col + "_PassingTypes")
                else:
                    new_stats_passing_types.append(col)
            headers = new_stats_passing_types

        if table_id == "stats_misc":
            new_stats_misc = []
            for col in headers:
                if col in ["Crs", "Int", "Lost", "Off", "TklW"]:
                    new_stats_misc.append(col + "_Misc")
                else:
                    new_stats_misc.append(col)
            headers = new_stats_misc
            
        # Extrair dados do corpo da tabela
        data = []
        for row in table.find('tbody').find_all('tr'):
            # Ignora linhas de subtítulos
            if row.get('class') and 'thead' in row.get('class'):
                continue
            cells = [td.text.strip() for td in row.find_all('td')]
            if cells:
                data.append(dict(zip(headers, cells)))

        # Converter para DataFrame
        df = pd.DataFrame(data)

        if index != 0:
            df.drop(["Squad", "Pos", "Nation", "90s", "Age", "Born"], axis=1, inplace=True)
        
        scrapped_data.append(df)
    return scrapped_data

def get_best_match(name, choices, threshold=90):
    match, score = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return match
    return None

def get_interactions(actions, game_id, player_before, player_after):
    game_actions = actions[actions['game_id'] == game_id]
    filtered = game_actions[game_actions['type_name'].isin(desired_actions)]
    sorted_data = filtered.sort_values(by=['period_id', 'time_seconds']).reset_index(drop=True)
