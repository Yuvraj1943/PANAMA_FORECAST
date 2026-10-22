"""Draws the two figures used in the presentation (title-slide chart and architecture diagram)."""
import pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp, matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
import sys
PAL = {'light': dict(BG='#FBFAF7', INK='#1B2430', MUTE='#6B7280', RULE='#D9D5CC', TEAL='#0F766E', BRICK='#B4452F', WHITE='#FFFFFF', BOX='#FFFFFF', BOXEDGE='#CFCAC0', GRID='#E8E5DD', PRIMTXT='#FFFFFF'),
       'dark': dict(BG='#10282A', INK='#F4EFE4', MUTE='#9DB4B0', RULE='#2D5558', TEAL='#3FC1B0', BRICK='#EE8366', WHITE='#17393B', BOX='#17393B', BOXEDGE='#2D5558', GRID='#1E4144', PRIMTXT='#10282A')}

def draw(variant):
    globals().update(PAL[variant])
    # ---- title-slide chart: one real week (21-27 Dec 2019) ----
    P = pd.read_csv('results/predictions_final.csv', parse_dates=['datetime']); d = P[P['fold'] == 'Week 51, Dec 2019']
    fig, ax = plt.subplots(figsize=(5.4, 3.5), dpi=240); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(d['datetime'], d['DEMAND'], color=INK, lw=2.4, label='Actual demand')
    ax.plot(d['datetime'], d['CND'], color=BRICK, lw=1.5, label='CND forecast')
    ax.plot(d['datetime'], d['XGBoost_paper'], color=TEAL, lw=1.6, label='XGBoost')
    for k in ('top', 'right'): ax.spines[k].set_visible(False)
    for k in ('left', 'bottom'): ax.spines[k].set_color(RULE)
    ax.grid(axis='y', color=GRID, lw=0.7); ax.tick_params(colors=MUTE, labelsize=7.5, length=0)
    ax.set_ylabel('Demand (MWh)', color=MUTE, fontsize=8); ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
    ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(pad=0.6); plt.savefig('figures/hero_week_' + variant + '.png', facecolor=BG); plt.close()

    # ---- architecture ----
    fig, ax = plt.subplots(figsize=(10, 3.3), dpi=240); fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.set_xlim(0, 100); ax.set_ylim(0, 33); ax.axis('off')
    cols = [('1  Data', 1, ['Panama demand, weather and\ncalendar (48,048 hours)', 'CND\'s official forecast\n(the benchmark)', '14 walk-forward weeks,\n72-hour information gap']),
            ('2  Features', 26, ['Demand 2, 3 and 4\nweeks earlier', '4-week moving average', 'Hour, weekday and\nholiday flags', 'Temperature (Tocumen)']),
            ('3  Models', 51, ['Persistence baseline', 'Linear Regression', 'XGBoost + Optuna\n(primary)', 'LSTM (secondary)']),
            ('4  Outputs', 76, ['Accuracy against CND\n(MAPE, MAE, RMSE)', 'SHAP explanations', 'Streamlit dashboard', '37 automated tests'])]
    W = 23
    for title, x, items in cols:
        ax.text(x, 30.6, title, ha='left', va='center', fontsize=10, fontweight='bold', color=INK)
        n = len(items); h = 5.2 if n == 4 else 6.9; gap = 1.0; top = 28.2
        for i, t in enumerate(items):
            y = top - (i + 1) * h - i * gap; prim = 'primary' in t
            ax.add_patch(mp.FancyBboxPatch((x, y), W, h, boxstyle='round,pad=0.0,rounding_size=0.5', fc=TEAL if prim else BOX, ec=TEAL if prim else BOXEDGE, lw=0.9))
            ax.text(x + W / 2, y + h / 2, t, ha='center', va='center', fontsize=7.8, color=PRIMTXT if prim else INK, fontweight='bold' if prim else 'normal', linespacing=1.15)
    for x0 in (24.3, 49.3, 74.3):
        ax.add_patch(FancyArrowPatch((x0, 15.2), (x0 + 1.5, 15.2), arrowstyle='-|>', mutation_scale=12, lw=1.4, color=INK))
    ax.text(50, 0.7, 'Runs weekly: rebuild features, retrain, forecast the next 168 hours, explain, compare with CND', ha='center', fontsize=8, style='italic', color=MUTE)
    plt.savefig('figures/architecture_' + variant + '.png', facecolor=BG, bbox_inches='tight', pad_inches=0.05); print('figures ok')


for v in ('light', 'dark'):
    draw(v)
