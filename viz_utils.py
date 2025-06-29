import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import matplotsoccer as mps

# Pré-processamento comum: adiciona end_x e end_y
def add_end_coords(df):
    df = df.copy()
    df['end_x'] = df['x'] + df['dx']
    df['end_y'] = df['y'] + df['dy']
    return df

def plot_action_counts(action_counts):
    plt.figure(figsize=(10, 6))
    action_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribuição dos Tipos de Ação')
    plt.xlabel('Tipo de Ação')
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_top_active_players(top_players):
    plt.figure(figsize=(10, 6))
    top_players.plot(kind='bar', color='lightgreen')
    plt.title('Top 10 Jogadores por Número de Ações')
    plt.xlabel('Jogador')
    plt.ylabel('Número de Ações')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_pass_heatmap(passes):
    n_x, n_y = 24, 16
    x_bins = np.linspace(0, 105, n_x + 1)
    y_bins = np.linspace(0, 68, n_y + 1)

    # Conta quantos passes começaram em cada quadrante
    heatmap, _, _ = np.histogram2d(passes['y'], passes['x'], bins=[y_bins, x_bins])

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(heatmap, cmap='Blues', origin='lower', extent=[0, 105, 0, 68], aspect='auto')

    for x in x_bins:
        ax.axvline(x, color='gray', linewidth=0.5)
    for y in y_bins:
        ax.axhline(y, color='gray', linewidth=0.5)

    ax.set_title('Mapa de Calor dos Passes')
    ax.set_xlabel('Comprimento do Campo')
    ax.set_ylabel('Largura do Campo')
    fig.colorbar(im, ax=ax, label='Número de Passes')
    plt.show()

def plot_action_sequences(action_sequences):
    plt.figure(figsize=(10, 6))
    action_sequences.plot(kind='bar', color='orange')
    plt.title('Tipo de Ação Imediatamente Antes do Gol')
    plt.xlabel('Tipo de Ação')
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_shot_heatmap(shots):
    # idêntico a pass_heatmap, mas para chutes
    n_x, n_y = 24, 16
    x_bins = np.linspace(0, 105, n_x + 1)
    y_bins = np.linspace(0, 68, n_y + 1)

    heatmap, _, _ = np.histogram2d(shots['y'], shots['x'], bins=[y_bins, x_bins])

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(heatmap, cmap='Reds', origin='lower', extent=[0, 105, 0, 68], aspect='auto')

    for x in x_bins:
        ax.axvline(x, color='gray', linewidth=0.5)
    for y in y_bins:
        ax.axhline(y, color='gray', linewidth=0.5)

    plt.title('Mapa de Calor de Finalizações')
    ax.set_xlabel('Comprimento do Campo')
    ax.set_ylabel('Largura do Campo')
    fig.colorbar(im, ax=ax, label='Número de Chutes')
    plt.show()

def plot_assists_heatmap(assists):
    # desenha setas de pass e recepção
    plt.figure(figsize=(12, 7))
    for _, row in assists.iterrows():
        plt.arrow(row['x'], row['y'],
                  row['end_x'] - row['x'], row['end_y'] - row['y'],
                  head_width=1, head_length=1, color='green', alpha=0.5)

    plt.title('Passes que Antecederam Chutes')
    plt.xlim(0, 105)
    plt.ylim(0, 68)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_buildup_last_events(spadl_dict, games_dict, last_n=10):
    for liga, gid in games_dict.items():
        df = add_end_coords(spadl_dict[liga])
        sub = df[df['game_id'] == gid].sort_values('time_seconds')

        # 1) filtragem de chutes e gols
        shots = sub[sub['type_name'].str.lower() == 'shot']
        goals = sub[sub['type_name'].str.lower() == 'goal']
        evt   = (goals.iloc[0] if not goals.empty else shots.iloc[0])

        # 2) pega eventos anteriores
        before      = sub[sub['time_seconds'] < evt['time_seconds']]
        last_events = before.tail(last_n)
        events_to_plot = pd.concat([last_events, pd.DataFrame([evt])], ignore_index=True)
        events_to_plot = events_to_plot.dropna(subset=['x', 'y'])

        ax = mps.field('green', figsize=8, show=False)

        xs = events_to_plot['x'].tolist()
        ys = events_to_plot['y'].tolist()

        ax.scatter(xs[:-1], ys[:-1], s=80, c='blue', zorder=3)
        for i in range(len(xs)-1):
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                    color='blue', linewidth=2, alpha=0.7, zorder=2)

        x0, y0 = evt['x'], evt['y']
        x1, y1 = evt['end_x'], evt['end_y']
        if pd.notna(x1) and pd.notna(y1):
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(color='red', width=2,
                                        headwidth=8, headlength=8),
                        zorder=4)
            ax.scatter([x1], [y1], s=150, marker='X',
                       c='red', zorder=5, label='Finalização')
        else:
            ax.scatter([x0], [y0], s=150, marker='X',
                       c='red', zorder=5, label='Shot')

        ax.set_title(
            f"{liga} • Partida {gid} • Últimos {last_n} eventos + "
            f"{'Gol' if evt['type_name'].lower()=='goal' else 'Shot'}"
        )
        ax.legend(loc='upper left', fontsize='small')
        plt.show()

def plot_attack_heatmap(spadl_dict, games_dict, bins=25):
    for liga, gid in games_dict.items():
        df = add_end_coords(spadl_dict[liga])
        sub = df[df['game_id'] == gid]
        shots = sub[sub['type_name'].str.lower() == 'shot']
        goals = sub[sub['type_name'].str.lower() == 'goal']
        evt = (goals.iloc[0] if not goals.empty else shots.iloc[0])
        team_act = sub[sub['team_id'] == evt['team_id']]

        ax = mps.field('green', figsize=8, show=False)

        hm = mps.count(team_act['x'], team_act['y'], n=bins, m=bins)
        hm = scipy.ndimage.gaussian_filter(hm, sigma=1)

        mps.heatmap(hm, cmap='Reds', linecolor='white', cbar=True, ax=ax)

        ax.text(
            0.02, 0.98,
            f"Liga: {liga}   Partida: {gid}",
            transform=ax.transAxes,
            ha='left', va='top',
            color='white',
            fontsize=12,
            backgroundcolor='black',
            alpha=0.6
        )

        ax.set_title(f"Heatmap de ataques (time {evt['team_id']})", pad=20)
        plt.show()