"""Overview map of recommendation behavior.

19 presentations (rows) x 4 intervention lists (columns). Each cell is a strip of
thin bars: the cell's options ranked by recommendation rate (share of the cell's open-
generation responses that recommend the option, where recommended = named without
being advised against), highest first, on a 0-100% scale shared by every strip.
Bar color is the option's mean rating in the structured arm (amber -3, gray 0,
teal +3); bars for options rated below 0 are hatched. The outlined first bar is the
option named beneath the strip. Data: data/cells.json.
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from cellutil import (COND_DISPLAY, CONTROLS, LIST_LABEL, LISTS, OUT, fmt_pct,  # noqa: E402
                      load_cells, ordered_conditions, short, sorted_options)
from figstyle import HAIR, INK, INK2, PANEL, T  # noqa: E402
from figstyle import apply_style, save  # noqa: E402

NBARS = 16            # ranks shown per strip
THRESH = 0.05         # "named by at least 5% of responses" for the printed summary (matches cells.py)

W = 7.0
LAB_R = 1.22          # right edge of the presentation-label column
GRID_L = 1.34
GRID_R = W - 0.06
COL_GAP = 0.14
ROW_H = 0.44          # row pitch (strip + top-option label)
STRIP_H = 0.26        # strip height = 100 %
HEAD_H = 0.58         # header band (list names + color key)
NOTE_H = 0.32         # bottom margin: clears the top-option label under the last strip
CTRL_GAP = 0.16       # extra space above the control rows

# Type sizing. Every explicit point size in the figure passes through ts(); the
# default (scale 1.0, no floor) reproduces the authored sizes exactly.
TYPE_SCALE = 1.0      # multiplier on every explicit type size
TYPE_MIN = 0.0        # floor in points applied after the multiplier


def ts(size):
    return max(size * TYPE_SCALE, TYPE_MIN)


# color-vision-safe diverging ramp: amber (-3) -> gray (0) -> teal (+3)
RATING_CMAP = LinearSegmentedColormap.from_list("rating", ["#D9822B", "#b5b5b5", "#4A8DB0"])
NEG_EDGE = "#8a5a1e"


def rating_color(o):
    m = o.get("rating_mean")
    if m is None:
        return "#d9d9d9"
    return RATING_CMAP(min(1.0, max(0.0, (float(m) + 3.0) / 6.0)))


def print_summary(cells, clinical, controls):
    rows = clinical + controls
    print("=== overview_map summary (recommendation rate = share of the cell's responses "
          "recommending the option) ===")
    print("%-16s %14s %14s %12s" % ("list", "median #>5%", "range #>5%", "median top"))
    for lst in LISTS:
        n_above, top = [], []
        for cond in rows:
            rates = [o["named_rate"] for _n, o in sorted_options(cells["%s|%s" % (cond, lst)])]
            n_above.append(sum(1 for r in rates if r >= THRESH))
            top.append(rates[0] if rates else 0.0)
        print("%-16s %14.0f %9d-%-4d %11.0f%%" % (
            lst, statistics.median(n_above), min(n_above), max(n_above),
            100 * statistics.median(top)))
    print("controls:")
    for cond in controls:
        for lst in LISTS:
            opts = sorted_options(cells["%s|%s" % (cond, lst)])
            rates = [o["named_rate"] for _n, o in opts]
            print("  %-22s %-16s %2d options >5%%; top %s %.0f%%" % (
                COND_DISPLAY[cond], lst, sum(1 for r in rates if r >= THRESH),
                short(opts[0][0]), 100 * rates[0]))
    print("negative-rated options with recommendation rate >= 5%:")
    for cond in rows:
        for lst in LISTS:
            for nm, o in sorted_options(cells["%s|%s" % (cond, lst)]):
                if o["named_rate"] >= THRESH and o.get("rating_mean") is not None and o["rating_mean"] < 0:
                    print("  %-22s %-16s %-28s named %5.1f%%  mean rating %+.2f" % (
                        COND_DISPLAY[cond], lst, nm, 100 * o["named_rate"], o["rating_mean"]))


def main():
    apply_style()
    plt.rcParams["hatch.linewidth"] = 0.35
    cells = load_cells()
    clinical, controls = ordered_conditions()
    rows = clinical + controls
    n = len(rows)

    H = HEAD_H + n * ROW_H + CTRL_GAP + NOTE_H
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    col_w = (GRID_R - GRID_L) / 4.0
    strip_w = col_w - COL_GAP
    pitch = strip_w / NBARS
    bar_w = pitch * 0.68
    top = H - HEAD_H

    def row_y(i):
        y = top - (i + 1) * ROW_H + (ROW_H - STRIP_H) / 2
        if rows[i] in CONTROLS:
            y -= CTRL_GAP
        return y

    # control band
    y_ctrl_top = row_y(len(clinical) - 1) - (ROW_H - STRIP_H) / 2 - CTRL_GAP / 2
    y_ctrl_bot = row_y(n - 1) - (ROW_H - STRIP_H) / 2 - 0.18   # room for the last row's option label
    ax.add_patch(Rectangle((0.06, y_ctrl_bot), W - 0.12, y_ctrl_top - y_ctrl_bot,
                           fc=PANEL, ec="none", zorder=0))

    # column headers
    y_head = top + 0.13
    for j, lst in enumerate(LISTS):
        x0 = GRID_L + j * col_w
        T(ax, x0, y_head, LIST_LABEL[lst], size=ts(9.2), fam="serif", weight=600, va="baseline")
        ax.plot([x0, x0 + strip_w], [top + 0.04, top + 0.04], color=INK2, lw=0.6,
                solid_capstyle="butt", zorder=1)

    # shared y-scale, drawn once
    xs, ys = LAB_R + 0.04, top + 0.04
    ax.plot([xs, xs], [ys, ys + STRIP_H], color=INK2, lw=0.5, zorder=2)
    for v in (0.0, 0.5, 1.0):
        ax.plot([xs - 0.025, xs], [ys + v * STRIP_H] * 2, color=INK2, lw=0.5)
    T(ax, xs - 0.04, ys + STRIP_H, "100%", size=ts(4.8), color=INK2, ha="right", va="center", fam="mono")
    T(ax, xs - 0.04, ys, "0", size=ts(4.8), color=INK2, ha="right", va="center", fam="mono")
    T(ax, xs - 0.30, ys + STRIP_H / 2, "recommendation rate", size=ts(5.6), color=INK2, ha="right", va="center")

    # color key
    kx, ky, kw, kh = GRID_R - 0.80, top + 0.36, 0.75, 0.06
    for q in range(30):
        ax.add_patch(Rectangle((kx + q * kw / 30, ky), kw / 30, kh, fc=RATING_CMAP(q / 29),
                               ec="none", zorder=2))
    T(ax, kx - 0.04, ky + kh / 2, "mean rating", size=ts(5.2), color=INK2, ha="right", va="center")
    for v, lab in ((0, "−3"), (0.5, "0"), (1, "+3")):
        T(ax, kx + v * kw, ky - 0.03, lab, size=ts(4.8), color=INK2, ha="center", va="top", fam="mono")

    # rows
    for i, cond in enumerate(rows):
        y0 = row_y(i)
        T(ax, LAB_R, y0 + STRIP_H * 0.42, COND_DISPLAY[cond], size=ts(7.8), color=INK,
          ha="right", va="center")
        for j, lst in enumerate(LISTS):
            x0 = GRID_L + j * col_w
            ax.plot([x0, x0 + strip_w], [y0, y0], color=HAIR, lw=0.6, solid_capstyle="butt", zorder=1)
            ax.plot([x0, x0 + strip_w], [y0 + STRIP_H / 2] * 2, color=HAIR, lw=0.4,
                    ls=(0, (1.2, 1.6)), zorder=1)
            opts = sorted_options(cells["%s|%s" % (cond, lst)])[:NBARS]
            for k, (_nm, o) in enumerate(opts):
                r = o["named_rate"]
                if r <= 0:
                    continue
                neg = o.get("rating_mean") is not None and float(o["rating_mean"]) < 0
                ax.add_patch(Rectangle(
                    (x0 + k * pitch + (pitch - bar_w) / 2, y0), bar_w, r * STRIP_H,
                    fc=rating_color(o),
                    ec=INK if k == 0 else (NEG_EDGE if neg else "none"),
                    lw=0.6 if k == 0 else (0.3 if neg else 0),
                    hatch="////" if neg else None, zorder=4 if k == 0 else 3))
            if opts and opts[0][1]["named_rate"] > 0:
                nm, o = opts[0]
                T(ax, x0, y0 - 0.035, "%s %s" % (short(nm), fmt_pct(o["named_rate"])),
                  size=ts(6.2), color=INK2, va="top")

    T(ax, 0.12, y_ctrl_top - 0.035, "watchful-waiting controls", size=ts(5.4), color=INK2,
      ha="left", va="top", fam="mono")

    # No in-figure footer: the caption carries the reading instructions.

    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "overview_map"))
    print_summary(cells, clinical, controls)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
