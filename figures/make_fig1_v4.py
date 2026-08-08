#!/usr/bin/env python
"""make_fig1_v4.py - Figure 1 method schematic, v4 layout.

Panel B redrawn for the replication round: the 40% row is the only row that
carries a claim and is shown as such, while the 0% row is visually demoted to
what it now is — a column evaluated only in the initial campaign, whose effective
sample size is capped by header combinatorics (paper §4.5). The derangement
control is annotated as having been re-run on the same 400 programs.

Original v3 layout preserved in make_fig1.py; this writes fig1_method_v4.*.

Larger canvas + larger type; the cross-lane training edge is an elbow routed
through empty space (no text/arrow crossings); all annotations sit clear of
every arrow path.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN = '#0072B2', '#E69F00', '#009E73'
BLUE_D, ORANGE_D, GREEN_D = '#005587', '#8a6100', '#00694f'
INK, SUB, MUT = '#1a1a1a', '#444444', '#8a8a8a'
LANE_O, LANE_B = '#fdf8ee', '#f2f7fb'

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'Inter', 'mathtext.fontset': 'stixsans',
                     'figure.dpi': 300, 'savefig.dpi': 300})

# Height compressed from 4.1in so the full-width figure with its full caption fits
# above section 3.1 rather than floating past it. Font sizes are in points and do
# not scale with the canvas, so only the layout tightens; the coordinates below
# were re-spaced where that tightening would have collided.
fig = plt.figure(figsize=(5.6, 3.2))
axA = fig.add_axes([0.0, 0.50, 1.0, 0.48]); axA.axis('off')
axB = fig.add_axes([0.0, -0.03, 1.0, 0.49]); axB.axis('off')
for ax in (axA, axB):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

def box(ax, x, y, w, h, text, ec='#b9b9b9', fc='white', fs=7.0, tc=INK, lw=1.2, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5,rounding_size=1.8',
                                fc=fc, ec=ec, lw=lw, mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            color=tc, linespacing=1.4,
            family='Liberation Mono' if mono else 'Inter')

def arrow(ax, x0, y0, x1, y1, color='#a0a0a0', lw=1.1, ls='-', cs=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', color=color,
                                 lw=lw, linestyle=ls, mutation_scale=8.5,
                                 shrinkA=1.5, shrinkB=1.5,
                                 connectionstyle=cs or 'arc3,rad=0'))

# ================= Panel A =================
axA.text(1, 99, 'A', fontsize=10, fontweight='bold', color=INK, va='top')
axA.text(6.5, 99, 'CADCON: header construction (training)  vs.  independent scoring (evaluation)',
         fontsize=7.6, fontweight='semibold', color=SUB, va='top')

axA.add_patch(Rectangle((1, 54), 98, 32, fc=LANE_O, ec='none', zorder=0))
axA.add_patch(Rectangle((1, 4), 98, 32, fc=LANE_B, ec='none', zorder=0))
axA.text(2.8, 83.5, 'HEADER CONSTRUCTION', fontsize=6.0, color=ORANGE_D,
         ha='left', va='center', fontweight='semibold', style='italic')
axA.text(2.8, 33.5, 'GENERATION & SCORING', fontsize=6.0, color=BLUE_D,
         ha='left', va='center', fontweight='semibold', style='italic')

# top lane boxes (y 54-76)
box(axA, 3, 58, 16, 20, 'DeepCAD\nJSON history', ec='#c9c9c9')
arrow(axA, 19.6, 68, 24.4, 68)
axA.text(22, 52.4, 'transpile', fontsize=5.8, color=MUT, ha='center')
box(axA, 25, 58, 17, 20, 'CadQuery\nprogram $P$', ec='#c9c9c9')
arrow(axA, 42.6, 68, 47.4, 68)
axA.text(45, 52.4, 'regex $\\phi$', fontsize=6.0, color=ORANGE_D, ha='center', fontweight='bold')
box(axA, 48, 58, 17, 20, 'intent $\\mathcal{I}$\n5 features', ec=ORANGE)
arrow(axA, 65.6, 68, 70.4, 68)
box(axA, 71, 58, 27, 20, '<CONSTRAINTS><CIRCLE>..\n# design intent: circle',
    ec=ORANGE, fs=5.5, mono=True, tc=SUB)
axA.text(84.5, 82.5, 'header $h(\\mathcal{I})$', fontsize=6.2, color=ORANGE_D, ha='center')

# bottom lane boxes (y 9-31)
box(axA, 3, 8, 22, 20, 'LoRA fine-tune M\non  $h(\\mathcal{I})+P$', ec=BLUE)
arrow(axA, 25.6, 18, 30.4, 18)
box(axA, 31, 8, 17, 20, 'completion\n(greedy)', ec=BLUE)
arrow(axA, 48.6, 18, 53.4, 18)
axA.text(51, 1.6, 'execute', fontsize=5.8, color=MUT, ha='center')
box(axA, 54, 8, 14, 20, 'B-rep\nsolid', ec='#c9c9c9')
arrow(axA, 68.6, 18, 73.4, 18)
box(axA, 74, 8, 24, 20, 'geometric assertions\nno code shared with $\\phi$', ec=GREEN, fs=6.4)

# cross-lane edges — elbow routed through empty space, labels clear of paths
# training edge: out of CadQuery's LEFT side, elbow down into LoRA's top
axA.plot([33.5, 33.5, 23], [57.2, 44.5, 44.5], color='#a0a0a0', lw=1.1,
         solid_joinstyle='miter', zorder=1)
arrow(axA, 23, 44.5, 23, 29.0)
axA.text(35.5, 45.5, '$h(\\mathcal{I})+P$  (training pair)', fontsize=5.6, color=MUT, ha='left')
axA.text(35.5, 38.5, 'eval: {0, 40}% prefix', fontsize=5.6, color=MUT, ha='left')
# target edge: dashed orange, label to its left
arrow(axA, 84.5, 56.5, 85.5, 29.5, color=ORANGE, lw=1.0, ls=(0, (3.5, 2.5)))
axA.text(82.5, 42.5, 'target only:\n$\\phi(GT)$', fontsize=5.6, color=ORANGE_D,
         ha='right', linespacing=1.4)

# ================= Panel B =================
axB.text(1, 96, 'B', fontsize=10, fontweight='bold', color=INK, va='top')
axB.text(6.5, 96, 'Pre-registered evaluation  +  derangement control, on one frozen sample',
         fontsize=7.6, fontweight='semibold', color=SUB, va='top')

# left: matrix
x0, y0, cw, chh = 8, 22, 16.5, 17
cols = ['correct', 'wrong', 'masked']
for j, c in enumerate(cols):
    axB.text(x0 + cw * j + (cw - 1.8) / 2, y0 + 2 * chh + 5.5, c, fontsize=6.6,
             ha='center', color=SUB, fontweight='medium')
for i, r in enumerate(['40% prefix', '0% prefix']):
    axB.text(x0 - 2, y0 + chh * (1 - i) + chh / 2 - 0.8, r, fontsize=6.6, ha='right',
             va='center', color=SUB if i == 0 else '#adadad',
             fontweight='medium' if i == 0 else 'normal')
for i in range(2):
    for j in range(3):
        live = (i == 0)                      # 只有 40% 行承载主张
        fc = '#e9f1f9' if live else '#fafafa'
        ec = '#9dbfd8' if live else '#dcdcdc'
        axB.add_patch(FancyBboxPatch((x0 + cw * j, y0 + chh * (1 - i)), cw - 1.8, chh - 1.8,
                                     boxstyle='round,pad=0.3,rounding_size=1.2',
                                     fc=fc, ec=ec, lw=0.9 if live else 0.6,
                                     ls='-' if live else (0, (2.2, 1.8)), zorder=2))
axB.text(x0 + 1.5 * cw - 0.9, y0 - 7.5, r'$\times$ {token, text}  $\times$ 3 seeds   +   unconditioned baseline',
         fontsize=6.0, ha='center', color=MUT)
axB.text(x0 + 1.5 * cw - 0.9, y0 - 14.0,
         'every claim in this paper: 400 deduplicated programs, 11 intent profiles',
         fontsize=6.0, ha='center', color=BLUE_D, fontweight='medium')
axB.text(x0 + 1.5 * cw - 0.9, y0 + 2 * chh + 13.5, 'header arms (M)', fontsize=7.0,
         ha='center', color=INK, fontweight='semibold')
# 0% 行的地位标注：放在矩阵下方注解区，避开右侧控制图的版面
axB.text(x0 + 1.5 * cw - 0.9, y0 - 20.5,
         '0% row: initial evaluation only — $\\leq$19 distinct prompts regardless of sample size (§4.5)',
         fontsize=5.6, ha='center', color='#9a9a9a', style='italic')

# right: derangement control — labels ABOVE/BELOW the arrow corridor
bx = 62
box(axB, bx, 60, 15, 21, 'M\ntrue pairs\n$(h(\\mathcal{I}_i),\\,P_i)$', ec=BLUE, fs=6.2)
box(axB, bx, 20, 15, 21, 'R\nderanged pairs\n$(h(\\mathcal{I}_j),\\,P_i)$', ec=GREEN, fs=6.2)
axB.text(bx + 7.5, 47.5, 'same marginal,\ncorrelation destroyed', fontsize=5.8,
         ha='center', color=MUT, style='italic', linespacing=1.35)
arrow(axB, bx + 15.6, 68, bx + 21.5, 56)
arrow(axB, bx + 15.6, 33, bx + 21.5, 45)
box(axB, bx + 21, 40, 14, 21, 'Test D\ninteraction\n$\\Delta_M-\\Delta_R$', ec=ORANGE, fs=6.2)
axB.text(bx + 18, 10.0, 'harm requires the learned header$\\,\\rightarrow\\,$program mapping?',
         fontsize=6.0, ha='center', color=SUB, style='italic')
axB.text(bx + 18, -3.0, 'same 400 programs, same indices,\nbyte-identical wrong headers',
         fontsize=5.6, ha='center', color=BLUE_D, linespacing=1.35)

fig.savefig(HERE / 'fig1_method_v4.png', bbox_inches='tight', pad_inches=0.02)
fig.savefig(HERE / 'fig1_method_v4.pdf', bbox_inches='tight', pad_inches=0.02)
bdir = HERE.parent / 'build' / 'figures'
if bdir.is_dir():
    fig.savefig(bdir / 'fig1_method_v4.pdf', bbox_inches='tight', pad_inches=0.02)
print('wrote fig1_method_v4.png/.pdf (+build/figures)')
