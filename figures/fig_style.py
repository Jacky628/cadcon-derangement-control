"""fig_style.py - shared visual language for the CADCON paper figures.

Inter type system, Okabe-Ito colors with stroke+tint usage, recessive grid,
rounded-top bars (flat base on the baseline), soft annotations.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Okabe-Ito anchors
BLUE, ORANGE, GREEN, VERM = '#0072B2', '#E69F00', '#009E73', '#D55E00'
# darker label variants (text on white must be darker than the mark)
BLUE_D, ORANGE_D, GREEN_D = '#005587', '#8a6100', '#00694f'
# tints for fills
BLUE_T, ORANGE_T, GREEN_T = '#d6e6f2', '#faeed4', '#d5efe6'
INK, SUB, MUT, FAINT = '#1a1a1a', '#444444', '#777777', '#b0b0b0'
GRID = '#ebebeb'

def apply():
    plt.rcParams.update({
        'font.family': 'Inter',
        'mathtext.fontset': 'stixsans',
        'font.size': 7.5, 'axes.titlesize': 8.5, 'axes.labelsize': 7.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.edgecolor': '#cccccc', 'axes.linewidth': 0.7,
        'xtick.color': '#cccccc', 'ytick.color': '#cccccc',
        'xtick.labelcolor': SUB, 'ytick.labelcolor': MUT,
        'text.color': INK, 'axes.labelcolor': SUB,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.facecolor': 'white',
    })

def grid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=0)

def rbar(ax, x, h, w, color, ec=None, hatch=None, r=0.018, zorder=3, alpha=1.0,
         shadow=False, sdx=0.045, sdy=0.0):
    """Rounded-top bar of EXACT height h: flat base at 0 (clipped), rounded top.

    The FancyBboxPatch spans (x-w/2, -rr) .. (x-w/2+w, h) with inset rounding rr,
    then a clip rectangle at y>=0 removes the rounded bottom, so the top edge sits
    exactly at h and the base is flat on the axis. With shadow=True a soft card
    shadow (offset sdx/sdy in data units) is laid underneath — a flat drop shadow,
    not a 3-D bar (perspective bars distort value reading and are avoided in print).
    """
    if h <= 0:
        return
    rr = min(r, h / 2)
    if shadow:
        sh = FancyBboxPatch(
            (x - w / 2 + sdx, -rr - 1e-6 + sdy), w, h + rr + 1e-6,
            boxstyle=f'round,pad=0,rounding_size={rr}', mutation_aspect=1,
            fc='#3a3a3a', ec='none', alpha=0.13, zorder=zorder - 1)
        ax.add_patch(sh)
        sh.set_clip_path(plt.Rectangle((x - w / 2 - 0.02, 0), w + 0.12, h + 1,
                                       transform=ax.transData))
    patch = FancyBboxPatch(
        (x - w / 2, -rr - 1e-6), w, h + rr + 1e-6,
        boxstyle=f'round,pad=0,rounding_size={rr}',
        mutation_aspect=1,
        fc=color, ec=ec or 'none', lw=0.9 if ec else 0,
        hatch=hatch, zorder=zorder, alpha=alpha)
    ax.add_patch(patch)
    patch.set_clip_path(plt.Rectangle((x - w / 2 - 0.02, 0), w + 0.04, h + 1,
                                      transform=ax.transData))

def save(fig, path_base):
    from pathlib import Path as _P
    fig.savefig(f'{path_base}.png', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{path_base}.pdf', bbox_inches='tight', pad_inches=0.02)
    bdir = _P(str(path_base)).parent.parent / 'build' / 'figures'
    if bdir.is_dir():
        name = _P(str(path_base)).name
        fig.savefig(bdir / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'wrote {path_base}.png/.pdf (+build/figures)')
