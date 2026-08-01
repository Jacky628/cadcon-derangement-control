#!/usr/bin/env python
"""make_figures.py - Figures 2-4 for the CADCON paper, from figures_data.json.

Visual language: fig_style (Inter, Okabe-Ito stroke+tint, rounded-top bars,
recessive grid). Numbers unchanged - read only from figures_data.json.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import fig_style as st
from fig_style import (BLUE, ORANGE, GREEN, BLUE_D, ORANGE_D, GREEN_D,
                       GREEN_T, INK, SUB, MUT, FAINT)

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / 'figures_data.json'))
st.apply()

# ================= Figure 2: seven-arm geom vs regex =================
t2g, t2r = D['table2']['geom'], D['table2']['regex']
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), sharey=True, constrained_layout=True)
for ax, tok, panel in ((axes[0], 'text', 'A'), (axes[1], 'token', 'B')):
    st.grid(ax)
    labels = ['baseline', 'correct', 'wrong', 'masked']
    geom = [t2g['baseline_p04']] + [t2g[f'{tok}_{a}_p04'] for a in labels[1:]]
    regex = [t2r['baseline_p04']] + [t2r[f'{tok}_{a}_p04'] for a in labels[1:]]
    x = np.arange(len(labels)); w = 0.34
    for xi, (g, r) in enumerate(zip(geom, regex)):
        st.rbar(ax, xi - w/2 - 0.015, g, w, BLUE, shadow=True)
        st.rbar(ax, xi + w/2 + 0.015, r, w, '#fdf6e9', ec=ORANGE, hatch='////', shadow=True)
        if g > 0.01:
            ax.text(xi - w/2 - 0.015, g + 0.015, f'{g:.2f}'.lstrip('0'), ha='center',
                    fontsize=6, color=BLUE_D, fontweight='medium')
        if r > 0.01:
            ax.text(xi + w/2 + 0.015, r + 0.015, f'{r:.2f}'.lstrip('0'), ha='center',
                    fontsize=6, color=ORANGE_D, fontweight='medium')
    ax.axhline(t2g['baseline_p04'], color=BLUE, ls=(0, (5, 4)), lw=0.8, alpha=0.65, zorder=2)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.62, len(labels) - 0.38)
    ax.set_title(f'{panel}   {tok} header', loc='left', fontweight='semibold', color=INK, pad=7)
    ax.set_ylim(0, 0.80)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
axes[0].set_ylabel('adherence (4-feature recall)')
axes[0].text(3.55, t2g['baseline_p04'] + 0.018, 'unconditioned baseline',
             fontsize=5.6, color=BLUE, ha='right', alpha=0.9)
# legend: manual, top-right of panel B
hx = 2.06
axes[1].add_patch(plt.Rectangle((hx - 0.16, 0.755), 0.24, 0.028, fc=BLUE, ec='none'))
axes[1].text(hx + 0.16, 0.769, 'geometry metric', fontsize=6.2, va='center', color=SUB)
axes[1].add_patch(plt.Rectangle((hx - 0.16, 0.700), 0.24, 0.028, fc='#fdf6e9',
                                ec=ORANGE, lw=0.8, hatch='////'))
axes[1].text(hx + 0.16, 0.714, 'regex metric', fontsize=6.2, va='center', color=SUB)
st.save(fig, HERE / 'fig2_seven_arms')

# ================= Figure 3: dissociation dumbbell =================
td = D['testd']
fig, ax = plt.subplots(figsize=(3.4, 2.55), constrained_layout=True)
st.grid(ax)
x = np.arange(3); off = 0.14
ax.axhline(0, color='#999999', lw=0.8, zorder=2)
for xi in x:
    yM, yR = td['drop_M'][xi], td['drop_R'][xi]
    ax.annotate('', xy=(xi + off, yR), xytext=(xi - off, yM),
                arrowprops=dict(arrowstyle='-|>', color='#c4c4c4', lw=1.0,
                                shrinkA=5.5, shrinkB=5.5, mutation_scale=7), zorder=2)
    # p label in a soft pill
    p = td['p'][xi]
    m, e = f'{p:.1e}'.split('e')
    ax.text(xi, 0.372, f'p = {m}' + r'$\times$10$^{' + str(int(e)) + '}$',
            ha='center', fontsize=6.2, color=SUB,
            bbox=dict(boxstyle='round,pad=0.28', fc='#f6f6f6', ec='none'))
ax.scatter(x - off, td['drop_M'], s=52, color=BLUE, marker='o', zorder=4,
           edgecolors='white', linewidths=1.1, label='M (standard)')
ax.scatter(x + off, td['drop_R'], s=48, color=GREEN, marker='s', zorder=4,
           edgecolors='white', linewidths=1.1, label='R (derangement control)')
for xi in x:
    ax.text(x[xi] - off - 0.08, td['drop_M'][xi], f"{td['drop_M'][xi]:.2f}",
            ha='right', va='center', fontsize=6, color=BLUE_D, fontweight='medium')
    ax.text(x[xi] + off + 0.08, td['drop_R'][xi], f"{td['drop_R'][xi]:.2f}",
            ha='left', va='center', fontsize=6, color=GREEN_D, fontweight='medium')
ax.set_xticks(x, [f'seed {s}' for s in range(3)])
ax.set_xlim(-0.55, 2.75)
ax.set_ylabel('correct → wrong adherence drop')
ax.set_ylim(-0.13, 0.42)
ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
ax.text(2.72, 0.322, 'Test D interaction\n(pratt, one-sided)', ha='right', fontsize=5.4,
        color=MUT, va='top', linespacing=1.4)
ax.legend(loc='lower left', frameon=False, handlelength=1.0, borderaxespad=0.2,
          bbox_to_anchor=(0.02, 0.02), labelcolor=SUB)
st.save(fig, HERE / 'fig3_dissociation')

# ================= Figure 4: output identity =================
ident = D['identity']
fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
st.grid(ax)
x = np.arange(3); w = 0.34
fr_M = [a / b for a, b in ident['M']]
fr_R = [a / b for a, b in ident['R']]
for xi in x:
    st.rbar(ax, xi - w/2 - 0.015, fr_M[xi], w, BLUE, r=0.012, shadow=True)
    st.rbar(ax, xi + w/2 + 0.015, fr_R[xi], w, GREEN_T, ec=GREEN, hatch='..', r=0.012, shadow=True)
    a, b = ident['M'][xi]
    ax.text(xi - w/2 - 0.015, fr_M[xi] + 0.022, f'{a}/{b}', ha='center', fontsize=6,
            color=BLUE_D, fontweight='medium')
    a, b = ident['R'][xi]
    ax.text(xi + w/2 + 0.015, fr_R[xi] + 0.022, f'{a}/{b}', ha='center', fontsize=6,
            color=GREEN_D, fontweight='medium')
ax.set_xticks(x, [f'seed {s}' for s in range(3)])
ax.set_xlim(-0.62, 2.62)
ax.set_ylabel('identical realized feature set\n(correct vs. wrong header)')
ax.set_ylim(0, 1.09)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
handles = [plt.Rectangle((0, 0), 1, 1, fc=BLUE, ec='none'),
           plt.Rectangle((0, 0), 1, 1, fc=GREEN_T, ec=GREEN, lw=0.8, hatch='..')]
ax.legend(handles, ['M (standard)', 'R (derangement control)'],
          loc='upper center', frameon=False, handlelength=1.1, ncols=2,
          bbox_to_anchor=(0.5, 1.16), columnspacing=1.4, labelcolor=SUB)
st.save(fig, HERE / 'fig4_output_identity')
print('done')
