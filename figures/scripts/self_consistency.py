"""Self-consistency of the rated models.

For each model, over its own list-mapped recommendations in the generation arm that it also
rated in the structured arm: the share it itself rates below zero. Filled dot: open ask only.
Hollow dot: both asks in the cell. Dot area is proportional to the model's recommended
interventions per response. Models ordered by the open-ask share. Colour = developer.
Data: data/models.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402

from cellutil import OUT, ROOT  # noqa: E402
from figstyle import F, HAIR, INK, INK2  # noqa: E402
from figstyle import apply_style, save  # noqa: E402

FAM_COLOR = {
    "Anthropic": "#B8845C", "OpenAI": "#1B3A5C", "Google": "#4A8DB0", "xAI": "#2d2c2b",
    "Mistral": "#D9822B", "Qwen": "#7B6BA8", "DeepSeek": "#5E8C61", "Moonshot": "#8a8880",
}
GRID = "#e6e6e6"

W, H = 7.0, 4.19
LEFT, RIGHT = 0.085, 0.985
# Absolute inches, so the key band and the rotated model names keep their spacing
# whatever the canvas height is. No in-figure footer: the caption carries it.
TOP_IN = 0.598        # canvas top -> top of the axes
AX_H_IN = 2.438       # axes height
TOP = (H - TOP_IN) / H
BOTTOM = (H - TOP_IN - AX_H_IN) / H


def fy(inches_from_top):
    """Figure fraction for a point `inches_from_top` below the top of the canvas."""
    return 1.0 - inches_from_top / H

# dot area (pt^2) from recommended interventions per response
def area(r):
    return 9 + 3.2 * r


def main():
    apply_style()
    models = json.load(open(os.path.join(ROOT, "data", "models.json")))["models"]
    rated = sorted(((m, v) for m, v in models.items() if v["rated"]),
                   key=lambda kv: (kv[1]["self_share_rated_below_zero_open"],
                                   kv[1]["self_share_rated_below_zero"]))

    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)

    for i, (m, v) in enumerate(rated):
        c = FAM_COLOR[v["family"]]
        yo = 100 * v["self_share_rated_below_zero_open"]
        yb = 100 * v["self_share_rated_below_zero"]
        ax.plot([i, i], [yo, yb], color=HAIR, lw=0.8, zorder=1)
        ax.scatter([i], [yb], s=area(v["rec_per_response"]), color="white", edgecolors=c,
                   linewidths=0.9, zorder=3)
        ax.scatter([i], [yo], s=area(v["rec_per_response_open"]), color=c, edgecolors="white",
                   linewidths=0.5, zorder=4)

    ax.set_xlim(-0.8, len(rated) - 0.2)
    ax.set_ylim(-0.6, 15.5)
    ax.set_xticks(range(len(rated)))
    ax.set_xticklabels([m for m, _ in rated], rotation=90, fontsize=6.0,
                       fontfamily=F["mono"], color=INK)
    for lbl, (m, v) in zip(ax.get_xticklabels(), rated):
        lbl.set_color(FAM_COLOR[v["family"]])
    ax.set_yticks([0, 5, 10, 15])
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(F["mono"])
        lbl.set_color(INK2)
    ax.tick_params(axis="y", labelsize=7.2, length=2.5)
    ax.tick_params(axis="x", length=0, pad=3.0)
    ax.spines["left"].set_color(HAIR)
    ax.spines["bottom"].set_color(HAIR)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("% of its own recommendations\nit rates below 0", fontsize=7.2, color=INK2,
                  fontfamily=F["sans"])

    # ------------------------------------------------------------------ key band
    # Drawn in figure coordinates in the empty strip above the plotting area, so it
    # cannot collide with any dot.
    kf = dict(transform=fig.transFigure, clip_on=False, zorder=5)
    ky = fy(0.221)      # marker row
    kyl = fy(0.451)     # size-label row
    tf = dict(transform=fig.transFigure, fontsize=6.4, fontfamily=F["sans"], va="center")

    ax.scatter([0.093], [ky], s=area(12), color=INK2, edgecolors="white", linewidths=0.5, **kf)
    fig.text(0.108, ky, "unconstrained ask", color=INK, **tf)
    ax.scatter([0.238], [ky], s=area(12), color="white", edgecolors=INK2, linewidths=0.9, **kf)
    fig.text(0.253, ky, "both asks in the cell", color=INK, **tf)

    fig.text(0.815, ky, "dot area: recommended interventions per response", color=INK2,
             ha="right", **tf)
    for j, r in enumerate((5, 15, 28)):
        x = 0.858 + j * 0.058
        ax.scatter([x], [ky], s=area(r), color="white", edgecolors=INK2, linewidths=0.8, **kf)
        fig.text(x, kyl, str(r), transform=fig.transFigure, fontsize=5.6, color=INK2,
                 fontfamily=F["mono"], ha="center", va="center")

    # ------------------------------------------------------------- layout check
    # The rotated model names are the tallest thing under the axis, and nothing may
    # leave the canvas.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = H * fig.dpi
    low = min(t.get_window_extent(rend).y0 for t in ax.get_xticklabels()) / px
    print("layout: lowest model name at %.4f of the canvas (%s)"
          % (low, "OK" if low > 0.002 else "OFF CANVAS"))

    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "self_consistency"))

    # ------------------------------------------------------------------ audit table
    print()
    print(f"{'model':<24} {'family':<10} {'open %':>7} {'both %':>7} {'rec/resp open':>14} "
          f"{'rec/resp':>9}")
    print("-" * 76)
    for m, v in rated:
        print(f"{m:<24} {v['family']:<10} "
              f"{100 * v['self_share_rated_below_zero_open']:7.2f} "
              f"{100 * v['self_share_rated_below_zero']:7.2f} "
              f"{v['rec_per_response_open']:14.2f} {v['rec_per_response']:9.2f}")


if __name__ == "__main__":
    main()
