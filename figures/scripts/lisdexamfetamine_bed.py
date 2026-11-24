"""Lisdexamfetamine for binge-eating disorder, model by model.

One cell of the grid, opened up. For each of the 41 generation models, the share of its
responses in the cell that recommend lisdexamfetamine, separately to a clinician, to a
caregiver and to a patient on the open ask (3 responses per footing).
The hairline joining the three marks is the footing gap for that model. Below the axis,
one square per rated model, colored by the model's own mean rating of the option in the
structured arm on the same amber-to-teal ramp the overview map uses.
Data: data/highlights.json (cells.lisdexamfetamine_bed.per_model).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from cellutil import MODEL_ORDER, OUT, ROOT  # noqa: E402
from figstyle import F, HAIR, INK, INK2, NAVY, TEAL  # noqa: E402
from figstyle import apply_style, save  # noqa: E402

CELL = "lisdexamfetamine_bed"
FOOTINGS = ("clinical", "caregiver", "patient")
FOOT_LABEL = {"clinical": "clinician", "caregiver": "caregiver", "patient": "patient"}
DEV_ORDER = ["Anthropic", "OpenAI", "Google", "xAI", "Mistral", "Qwen", "DeepSeek", "Moonshot"]

GRID = "#e6e6e6"
RATING_CMAP = LinearSegmentedColormap.from_list("rating", ["#D9822B", "#b5b5b5", "#4A8DB0"])
NEG_EDGE = "#8a5a1e"

W, H = 7.0, 3.50
LEFT, RIGHT = 0.105, 0.985
# Absolute inches, so the key band and the label rows keep their spacing whatever the
# canvas height is. No in-figure footer: the caption carries it.
TOP_IN = 0.437        # canvas top -> top of the axes
AX_H_IN = 1.4934      # axes height
TOP = (H - TOP_IN) / H
BOTTOM = (H - TOP_IN - AX_H_IN) / H


def fy(inches_from_top):
    """Figure fraction for a point `inches_from_top` below the top of the canvas."""
    return 1.0 - inches_from_top / H


def fh(inches):
    """Figure-fraction height for a span of `inches`."""
    return inches / H


DEV_GAP = 2.4          # blank slots between developer blocks, in model-slot units
MS = 18                # circle marker area (pt^2)
SQ = 30                # rating square marker area (pt^2)
SQ_Y = -0.058          # square row center, in axes fraction below the axis line


def rating_color(m):
    return RATING_CMAP(min(1.0, max(0.0, (float(m) + 3.0) / 6.0)))


def main():
    apply_style()
    plt.rcParams["hatch.linewidth"] = 0.35
    hl = json.load(open(os.path.join(ROOT, "data", "highlights.json")))
    pm = hl["cells"][CELL]["per_model"]

    # x positions: one slot per model, a gap between developer blocks
    xs, blocks, x = {}, [], 0.0
    for dev in DEV_ORDER:
        members = [m for m in MODEL_ORDER if pm[m]["family"] == dev]
        for m in members:
            xs[m] = x
            x += 1.0
        blocks.append((dev, xs[members[0]], xs[members[-1]]))
        x += DEV_GAP
    span = x - DEV_GAP

    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)

    # ------------------------------------------------------------------ marks
    for m in MODEL_ORDER:
        v, xi = pm[m], xs[m]
        ys = [100 * v["named_by_ask_footing"]["open"][f] for f in FOOTINGS]
        ax.plot([xi, xi], [min(ys), max(ys)], color=HAIR, lw=0.8, zorder=2)
        ax.scatter([xi], [ys[0]], s=MS, color=NAVY, edgecolors="none", zorder=3)
        ax.scatter([xi], [ys[1]], s=MS, color=TEAL, edgecolors="none", zorder=4)
        ax.scatter([xi], [ys[2]], s=MS, facecolors="none", edgecolors=INK2, linewidths=0.8,
                   zorder=5)

    ax.set_xlim(-0.7, span + 0.7)
    ax.set_ylim(-4, 104)
    ax.set_xticks([xs[m] for m in MODEL_ORDER])
    ax.set_xticklabels(MODEL_ORDER, rotation=90, fontsize=5.8, fontfamily=F["mono"], color=INK)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50%", "100%"])
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(F["mono"])
        lbl.set_color(INK2)
    ax.tick_params(axis="y", labelsize=7.2, length=2.5)
    ax.tick_params(axis="x", length=0, pad=14.0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(HAIR)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("% of responses\nrecommending it", fontsize=7.2, color=INK2,
                  fontfamily=F["sans"], linespacing=1.4)

    # ------------------------------------------------------------------ rating squares
    # x in data units, y in axes fraction: the row sits between the axis line and the
    # model names. Square markers stay square whatever the axes aspect is.
    tb = blended_transform_factory(ax.transData, ax.transAxes)
    for m in MODEL_ORDER:
        r = pm[m]["rating_mean"]
        if r is None:
            continue
        neg = float(r) < 0
        ax.scatter([xs[m]], [SQ_Y], s=SQ, marker="s", color=rating_color(r),
                   edgecolors=NEG_EDGE if neg else "none", linewidths=0.3 if neg else 0,
                   hatch="////" if neg else None, transform=tb, clip_on=False, zorder=6)
    row_lab = ax.text(-0.95, SQ_Y, "its own rating", fontsize=6.2, color=INK2,
                      fontfamily=F["sans"], ha="right", va="center", transform=tb,
                      clip_on=False)

    # ------------------------------------------------------------------ developer names
    # One name per block, on a shared baseline just under the longest model name. The row
    # is measured rather than guessed, because the model names differ a lot in length.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = H * fig.dpi
    lab_bot = min(t.get_window_extent(rend).y0 for t in ax.get_xticklabels()) / px
    dev_y = lab_bot - fh(0.0988)
    devs = [ax.text((x0 + x1) / 2, dev_y, dev, transform=blended_transform_factory(
            ax.transData, fig.transFigure), fontsize=6.4, color=INK2, fontfamily=F["sans"],
            ha="center", va="baseline", clip_on=False) for dev, x0, x1 in blocks]

    # ------------------------------------------------------------------ key band
    # Figure coordinates, in the empty strip above the plotting area, so nothing in the
    # key can collide with a mark.
    kf = dict(transform=fig.transFigure, clip_on=False, zorder=7)
    tf = dict(transform=fig.transFigure, fontsize=6.4, fontfamily=F["sans"], va="center")
    ky = fy(0.201)

    ax.scatter([0.118], [ky], s=MS, color=NAVY, edgecolors="none", **kf)
    fig.text(0.132, ky, FOOT_LABEL["clinical"], color=INK, **tf)
    ax.scatter([0.202], [ky], s=MS, color=TEAL, edgecolors="none", **kf)
    fig.text(0.216, ky, FOOT_LABEL["caregiver"], color=INK, **tf)
    ax.scatter([0.290], [ky], s=MS, facecolors="none", edgecolors=INK2, linewidths=0.8, **kf)
    fig.text(0.304, ky, FOOT_LABEL["patient"], color=INK, **tf)

    kx, kw, kh = 0.800, 0.150, fh(0.0646)
    for q in range(30):
        ax.add_patch(Rectangle((kx + q * kw / 30, ky - kh / 2), kw / 30 + 0.0008, kh,
                               fc=RATING_CMAP(q / 29), ec="none", **kf))
    fig.text(kx - 0.012, ky, "mean rating", color=INK2, ha="right", **tf)
    for v, lab in ((0.0, "−3"), (0.5, "0"), (1.0, "+3")):
        fig.text(kx + v * kw, ky - kh / 2 - fh(0.0532), lab, transform=fig.transFigure,
                 fontsize=5.6, color=INK2, fontfamily=F["mono"], ha="center", va="top")

    # ------------------------------------------------------------------ layout check
    # Nothing may leave the canvas or run into the row below it.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def box(a):
        b = a.get_window_extent(rend)
        return (b.x0 / (W * fig.dpi), b.y0 / px, b.x1 / (W * fig.dpi), b.y1 / px)

    bad = []
    for nm, a in [("rating row label", row_lab)] + \
                 [("dev " + d, t) for (d, _a, _b), t in zip(blocks, devs)] + \
                 [("tick " + m, t) for m, t in zip(MODEL_ORDER, ax.get_xticklabels())]:
        x0, y0, x1, y1 = box(a)
        if x0 < 0.002 or x1 > 0.998 or y0 < 0.002 or y1 > 0.998:
            bad.append(f"{nm} outside canvas {x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f}")
    for (d0, t0), (d1, t1) in zip(list(zip(blocks, devs))[:-1], list(zip(blocks, devs))[1:]):
        if box(t0)[2] + 0.004 > box(t1)[0]:
            bad.append(f"developer names {d0[0]}/{d1[0]} too close "
                       f"({box(t1)[0] - box(t0)[2]:+.4f})")
    dy0 = min(box(t)[1] for t in devs)
    ly0 = min(box(t)[1] for t in ax.get_xticklabels())
    dy1 = max(box(t)[3] for t in devs)
    if dy1 + 0.004 > ly0:
        bad.append(f"dev row {dy0:.3f}-{dy1:.3f} runs into labels {ly0:.3f}")
    print("layout: " + ("OK" if not bad else "; ".join(bad)))

    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "lisdexamfetamine_bed"))

    # ------------------------------------------------------------------ audit table
    cell = hl["cells"][CELL]
    print()
    print(f"{CELL}: cell named {cell['named_rate']:.1%}, rating {cell['rating_mean']:+.2f}")
    print(f"{'model':<24} {'developer':<10} {'clin':>6} {'care':>6} {'pat':>6} {'rating':>7}")
    print("-" * 64)
    for m in MODEL_ORDER:
        v = pm[m]
        r = f"{v['rating_mean']:+.2f}" if v["rating_mean"] is not None else "-"
        print(f"{m:<24} {v['family']:<10} "
              + " ".join(f"{100 * v['named_by_ask_footing']['open'][f]:5.0f}%" for f in FOOTINGS)
              + f" {r:>7}")


if __name__ == "__main__":
    main()
