import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import matplotsoccer as mps

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
    heatmap, _, _ = np.histogram2d(passes['start_y'], passes['start_x'], bins=[y_bins, x_bins])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(heatmap, cmap='Blues', origin='lower', extent=[0, 105, 0, 68], aspect='auto')
    
    # Adiciona a grade
    for x in x_bins:
        ax.axvline(x, color='gray', linewidth=0.5)
    for y in y_bins:
        ax.axhline(y, color='gray', linewidth=0.5)
    
    # Rótulos e título
    ax.set_title('Mapa de Calor dos Passes')
    ax.set_xlabel('Comprimento do Campo')
    ax.set_ylabel('Largura do Campo')
    fig.colorbar(im, ax=ax, label='Número de Passes')
    plt.show()
    
def plot_action_sequences(action_sequences):
    plt.figure(figsize=(10, 6))
    action_sequences.plot(kind='bar', color='orange')
    plt.title('Tipo de Ação Imediatamente Antes do Gol (feito a partir de um chute)')
    plt.xlabel('Tipo de Ação')
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_shot_heatmap(shots):
    n_x, n_y = 24, 16
    x_bins = np.linspace(0, 105, n_x + 1)
    y_bins = np.linspace(0, 68, n_y + 1)
    
    # Conta quantos passes começaram em cada quadrante
    heatmap, _, _ = np.histogram2d(shots['start_y'], shots['start_x'], bins=[y_bins, x_bins])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(heatmap, cmap='Reds', origin='lower', extent=[0, 105, 0, 68], aspect='auto')
    
    # Adiciona a grade
    for x in x_bins:
        ax.axvline(x, color='gray', linewidth=0.5)
    for y in y_bins:
        ax.axhline(y, color='gray', linewidth=0.5)
    
    # Rótulos e título
    plt.title('Mapa de Calor de Finalizações')
    ax.set_xlabel('Comprimento do Campo')
    ax.set_ylabel('Largura do Campo')
    fig.colorbar(im, ax=ax, label='Número de Chutes')
    plt.show()
    
def plot_assists_heatmap(assists):
    plt.figure(figsize=(12, 7))
    for _, row in assists.iterrows():
        plt.arrow(row['start_x'], row['start_y'], 
                  row['end_x'] - row['start_x'], row['end_y'] - row['start_y'], 
                  head_width=1, head_length=1, color='green', alpha=0.5)
    
    plt.title('Passes que Antecederam Chutes')
    plt.xlim(0, 105)
    plt.ylim(0, 68)
    plt.gca().set_aspect('equal')
    plt.show()
        
def plot_buildup_last_events(spadl_dict, games_dict, last_n=10):
    import pandas as pd

    for liga, gid in games_dict.items():
        df  = spadl_dict[liga]
        sub = df[df['game_id']==gid].sort_values('time_seconds')

        # 1) escolhe o evento final
        shots = sub[sub['type_name'].str.lower()=='shot']
        goals = shots[shots['result_name'].str.lower()=='success']
        evt   = goals.iloc[0] if not goals.empty else shots.iloc[0]

        # 2) pega só os eventos antes e limita aos últimos last_n
        before      = sub[sub['time_seconds'] < evt['time_seconds']]
        last_events = before.tail(last_n)
        events_to_plot = pd.concat([last_events, pd.DataFrame([evt])], ignore_index=True)
        events_to_plot = events_to_plot.dropna(subset=['start_x','start_y'])

        # 3) desenha o campo e obtém ax
        ax = mps.field('green', figsize=8, show=False)

        # 4) extrai coordenadas
        xs = events_to_plot['start_x'].tolist()
        ys = events_to_plot['start_y'].tolist()

        # 5) plota últimos eventos (círculos) e conecta com linhas
        ax.scatter(xs[:-1], ys[:-1], s=80, c='blue', zorder=3)
        for i in range(len(xs)-1):
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                    color='blue', linewidth=2, alpha=0.7, zorder=2)

        # 6) plot da finalização
        x0, y0 = evt['start_x'], evt['start_y']
        x1, y1 = evt.get('end_x', None), evt.get('end_y', None)
        if pd.notna(x1) and pd.notna(y1):
            ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
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
            f"{'Gol' if evt['result_name'].lower()=='success' else 'Shot'}"
        )
        ax.legend(loc='upper left', fontsize='small')
        plt.show()

def plot_attack_heatmap(spadl_dict, games_dict, bins=25):
    """
    Para cada liga e partida em games_dict:
     - identifica o team_id do shot/gol
     - desenha um heatmap (suavizado) das posições de todas as ações
       desse time no campo
     - adiciona um label no canto com liga e partida
    """

    for liga, gid in games_dict.items():
        df       = spadl_dict[liga]
        sub      = df[df['game_id']==gid]
        shots    = sub[sub['type_name'].str.lower()=='shot']
        goals    = shots[shots['result_name'].str.lower()=='success']
        evt      = goals.iloc[0] if not goals.empty else shots.iloc[0]
        team_act = sub[sub['team_id']==evt['team_id']]

        # desenha o campo e obtém ax
        ax = mps.field('green', figsize=8, show=False)

        # gera e suaviza o heatmap
        hm = mps.count(team_act['start_x'], team_act['start_y'], n=bins, m=bins)
        hm = scipy.ndimage.gaussian_filter(hm, sigma=1)

        # plota o heatmap sobre o campo
        mps.heatmap(hm, cmap='Reds', linecolor='white', cbar=True, ax=ax)

        # label em coordenadas relativas (0 a 1)
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