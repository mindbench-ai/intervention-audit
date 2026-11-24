"""Catalogue -- one figure per (presentation, intervention list) cell.

76 cells: 19 presentations x 4 intervention lists. Each page is one cell, one row per
option, sorted by recommendation rate. A row carries three aligned readings of the same
option:

  * the option name;
  * open generation, as a paired bar: the recommendation rate (share of the cell's
    open-generation responses that recommend the option) above, the advised-against rate
    below, with three small marks giving the recommendation rate within each
    conversational footing;
  * structured rating, as the distribution of that option's ratings over the seven
    points -3 to +3, with a caret at the mean.

Descriptive only: the two elicitation arms sit side by side as two observations, and
nothing is pooled across cells. The recommendation-rate scale is set per cell and is
printed in the header, so scales differ between cells; the rating axis is identical in
every row of every cell.

Data: data/cells.json. Style: figstyle.py, labels: cellutil.py.

Outputs (figures/out/catalogue/):
    catalogue_<NN>_<condition>_<list>.pdf / .png    for NN = 01..76
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

from cellutil import (COND_DISPLAY, CONTROLS, LIST_LABEL, LISTS, OUT,  # noqa: E402
                      fmt_pct, load_cells, ordered_conditions, short, sorted_options)
from figstyle import F, HAIR, INK, INK2, NAVY, TEAL  # noqa: E402
from figstyle import ORANGE, apply_style, save  # noqa: E402

OUTDIR = os.path.join(OUT, "catalogue")

# The neuromodulation list is labeled "Procedures" in every figure. cellutil
# already says so; asserted here so a change there cannot silently relabel these pages.
LIST_LABEL = dict(LIST_LABEL)
LIST_LABEL["neuromodulation"] = "Procedures"

# ---------------------------------------------------------------- display rules
MIN_RATE = 0.02   # show every option recommended in >= 2% of the cell's responses ...
EXTRA = 6         # ... plus this many of the next-ranked ones
MAX_ROWS = 16     # hard cap
MIN_ROWS = 8      # pad with never-recommended options up to this many rows
X_STEP = 0.10     # the rate scale rounds up to the next 10% ...
X_FLOOR = 0.20    # ... and is never tighter than 0-20%

# ---------------------------------------------------------------- color
REC = "#2f7d5c"        # recommendation rate (figstyle GREEN, darkened for hairline bars)
ADV = ORANGE           # advised-against rate
RATING_CMAP = LinearSegmentedColormap.from_list("rating", ["#D9822B", "#b5b5b5", "#4A8DB0"])
FOOTINGS = ("clinical", "caregiver", "patient")
FOOT_LABEL = {"clinical": "clinician", "caregiver": "caregiver", "patient": "patient"}
RULE = "#c9c7c3"
ROW_HAIR = "#eeedeb"
TRACK = "#f0efed"

# ---------------------------------------------------------------- geometry (inches)
W = 7.0
NAME_L = 0.10          # left margin / left edge of the option-name column
NAME_R = 1.40          # right edge of the option-name column
BAR_L = 1.52           # bar track
BAR_R = 4.28
VAL_R = 4.70           # right-aligned value column
HIST_L = 5.00
HIST_R = 6.90
ROW_H = 0.36
PAD_GAP = 0.24         # extra gap holding the "never recommended" rule + label
TOP_H = 1.56           # title, column headers, keys

NAME_SIZE = 7.2
NAME_MIN_SIZE = 6.2
FOOT_SIZE = 6.0        # footnote type size
MS = 7                 # footing marker area (pt^2)


# ------------------------------------------------------------------ text metrics

_FP_CACHE = {}


def _fp(size, fam="sans", weight="normal", style="normal"):
    key = (size, fam, weight, style)
    if key not in _FP_CACHE:
        _FP_CACHE[key] = FontProperties(family=F[fam], size=size, weight=weight,
                                        style=style)
    return _FP_CACHE[key]


_W_CACHE = {}


def text_w(s, size, fam="sans", weight="normal", style="normal"):
    """Rendered width of `s` in inches, without needing a live renderer."""
    if not s:
        return 0.0
    key = (s, size, fam, weight, style)
    if key not in _W_CACHE:
        prop = _fp(size, fam, weight, style)
        _W_CACHE[key] = TextPath((0, 0), s, prop=prop).get_extents().width / 72.0
    return _W_CACHE[key]


def fit_size(s, avail, size, minsize, fam="sans", weight="normal", style="normal"):
    """Largest size <= `size` (floor `minsize`) at which `s` fits `avail` inches."""
    while size > minsize and text_w(s, size, fam, weight, style) > avail:
        size -= 0.1
    return size


def fit_label(s, avail, size=NAME_SIZE, minsize=NAME_MIN_SIZE):
    """Shrink then, only as a last resort, clip `s` to fit `avail` inches."""
    size = fit_size(s, avail, size, minsize)
    if text_w(s, size) <= avail:
        return s, size
    while len(s) > 2 and text_w(s.rstrip() + ".", size) > avail:
        s = s[:-1]
    return s.rstrip() + ".", size


def wrap_measured(text, avail, size):
    """Greedy word wrap to lines no wider than `avail` inches."""
    out, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if line and text_w(trial, size) > avail:
            out.append(line)
            line = word
        else:
            line = trial
    if line:
        out.append(line)
    return out


def T(ax, x, y, s, size=7, color=INK, ha="left", va="center", fam="sans",
      weight="normal", style="normal", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontfamily=F[fam], fontweight=weight, fontstyle=style, **kw)


# ------------------------------------------------------------------ display rules


def select_rows(cell):
    """Rows for one cell, plus the index at which never-recommended padding starts.

    Every option with a recommendation rate of at least MIN_RATE, plus the next EXTRA
    by rank, capped at MAX_ROWS. If that comes to fewer than MIN_ROWS the list is
    extended to MIN_ROWS, which may pull in options that were never recommended at all
    (rate 0); those are shown with an empty bar under a dashed rule.
    """
    ranked = sorted_options(cell)
    n_main = sum(1 for _o, v in ranked if v["named_rate"] >= MIN_RATE)
    n = min(MAX_ROWS, n_main + EXTRA)
    n = min(max(n, MIN_ROWS), len(ranked))
    rows = ranked[:n]
    pad_at = next((i for i, (_o, v) in enumerate(rows) if v["named_rate"] <= 0), None)
    if pad_at == 0:      # degenerate: nothing in the cell was ever recommended
        pad_at = None
    return rows, pad_at


def x_scale(rows):
    """Per-cell max over the bars and the footing marks, up to the next 10%, never
    tighter than 20%."""
    raw = max(
        max(v["named_rate"] for _o, v in rows),
        max(v["advised_rate"] for _o, v in rows),
        max(max(v["named_by_footing"].values()) for _o, v in rows),
    )
    steps = int(raw / X_STEP) + (0 if abs(raw / X_STEP - round(raw / X_STEP)) < 1e-9 else 1)
    return max(X_FLOOR, round(steps * X_STEP, 10))


def axis_ticks(xmax_pct):
    """Evenly spaced ticks landing exactly on 0 and on the cell's max.

    The largest step that divides the max and leaves at most 7 gaps, so no cell ends
    with a crowded pair at the right-hand end.
    """
    cand = [s for s in (50, 30, 25, 20, 10)
            if xmax_pct % s == 0 and 2 <= xmax_pct // s <= 7]
    if not cand:
        return [0, xmax_pct]
    step = min(cand, key=lambda s: (abs(xmax_pct // s - 4), -s))
    return list(range(0, xmax_pct + 1, step))


def cell_title(cond, lst):
    title = "%s / %s" % (COND_DISPLAY[cond], LIST_LABEL[lst])
    if cond in CONTROLS:
        return title + " (watchful-waiting control)", True
    return title, False


def ordered_cells():
    """Presentations A-Z by display name, controls last; fixed list order."""
    clinical, controls = ordered_conditions()
    return [(c, l) for c in list(clinical) + list(controls) for l in LISTS]


def stem_name(cond, lst, n):
    return "catalogue_%02d_%s_%s" % (n, cond, lst)


def rating_color(value):
    return RATING_CMAP(min(1.0, max(0.0, (float(value) + 3.0) / 6.0)))


# --------------------------------------------------------------------- the layout


def render_cell(cond, lst, cell):
    """One catalogue cell as a standalone matplotlib figure."""
    rows, pad_at = select_rows(cell)
    xmax = x_scale(rows)
    xmax_pct = int(round(100 * xmax))
    title, italic = cell_title(cond, lst)
    n_resp = cell["n_responses"]

    rating_tot = sorted(v["rating_n"] for _o, v in rows)
    rating_mid = rating_tot[len(rating_tot) // 2]

    # ---- footer text (the display rules, spelled out on every page) ------------
    foot = [
        "Rows: every option with a recommendation rate of at least 2%% of this cell's "
        "%d open-generation responses, plus the next %d by rank (cap %d rows, minimum "
        "%d)." % (n_resp, EXTRA, MAX_ROWS, MIN_ROWS),
        "Bars: recommendation rate above, advised-against rate below, both as a share "
        "of all %d responses; the marks above the upper bar are the recommendation "
        "rate within each conversational footing. The 0 to %d%% scale is set per cell "
        "(rounded up to the next 10%%, never below 20%%), so scales differ between "
        "cells." % (n_resp, xmax_pct),
        "Histogram: this option's structured ratings (about %d of them), normalised "
        "within the row; the caret marks the mean and the axis is identical in every "
        "row. Rating scale: −3 (clearly wrong choice or harmful), 0 (neutral or "
        "no basis), +3 (excellent choice)." % rating_mid,
    ]
    if pad_at is not None:
        foot.insert(1, "Rows under the dashed rule were never recommended in open "
                       "generation in this cell; they only fill out the row count.")

    avail_foot = HIST_R - NAME_L
    foot_lines = []
    for para in foot:
        foot_lines.extend(wrap_measured(para, avail_foot, FOOT_SIZE))

    # ---- geometry --------------------------------------------------------------
    bot_h = 0.42 + 0.135 * len(foot_lines) + 0.16   # axis ticks + footnote block
    gap = PAD_GAP if pad_at is not None else 0.0
    H = TOP_H + len(rows) * ROW_H + gap + bot_h

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    def bx(rate):
        return BAR_L + (BAR_R - BAR_L) * (rate / xmax)

    # ---- title -----------------------------------------------------------------
    tsize = fit_size(title, W - 2 * NAME_L, 10.5, 8.2, fam="serif", weight=600,
                     style="italic" if italic else "normal")
    T(ax, NAME_L, H - 0.30, title, size=tsize, fam="serif", weight=600,
      style="italic" if italic else "normal", va="baseline")
    sub = ("what open generation recommends, and how the structured arm rates the same "
           "options")
    T(ax, NAME_L, H - 0.50, sub, size=fit_size(sub, W - 2 * NAME_L, 7.2, 6.2),
      color=INK2, va="baseline")

    # ---- column headers --------------------------------------------------------
    hdr_y = H - 0.86
    hdr2 = hdr_y - 0.16
    key1 = hdr_y - 0.36
    key2 = hdr_y - 0.52
    T(ax, NAME_R, hdr_y, "option", size=8.2, fam="serif", weight=600, ha="right",
      va="baseline")
    T(ax, BAR_L, hdr_y, "Open generation", size=8.2, fam="serif", weight=600,
      va="baseline")
    T(ax, BAR_L, hdr2, "recommendation rate and advised-against rate", size=6.6,
      color=INK2, va="baseline")
    T(ax, HIST_L, hdr_y, "Structured rating", size=8.2, fam="serif", weight=600,
      va="baseline")
    T(ax, HIST_L, hdr2, "distribution, −3 to +3", size=6.6, color=INK2,
      va="baseline")

    # stance key
    kx = BAR_L
    for color, label in ((REC, "recommendation rate"), (ADV, "advised against")):
        ax.add_patch(Rectangle((kx, key1 - 0.005), 0.09, 0.055, fc=color, ec="none",
                               zorder=3))
        T(ax, kx + 0.115, key1, label, size=6.2, color=INK2, va="baseline")
        kx += 0.115 + text_w(label, 6.2) + 0.18
    T(ax, VAL_R, key1, "0 to %d%% scale" % xmax_pct, size=6.2, color=INK2, ha="right",
      va="baseline")
    T(ax, HIST_L, key1, "height = share of about %d ratings" % rating_mid, size=6.2,
      color=INK2, va="baseline")

    # footing key
    kx = BAR_L
    T(ax, kx, key2, "by footing:", size=6.2, color=INK2, va="baseline")
    kx += text_w("by footing:", 6.2) + 0.12
    for f in FOOTINGS:
        _foot_mark(ax, kx, key2 + 0.022, f)
        T(ax, kx + 0.075, key2, FOOT_LABEL[f], size=6.2, color=INK2, va="baseline")
        kx += 0.075 + text_w(FOOT_LABEL[f], 6.2) + 0.15

    grid_top = H - TOP_H
    ax.plot([NAME_L, HIST_R], [grid_top + 0.06] * 2, color=RULE, lw=0.8,
            solid_capstyle="butt")

    # ---- rows ------------------------------------------------------------------
    hist_w = HIST_R - HIST_L
    bin_w = hist_w / 7.0
    name_avail = NAME_R - NAME_L
    for i, (name, v) in enumerate(rows):
        shift = gap if (pad_at is not None and i >= pad_at) else 0.0
        yc = grid_top - (i + 0.5) * ROW_H - shift
        if i and i != pad_at:
            ax.plot([NAME_L, HIST_R], [yc + ROW_H / 2] * 2, color=ROW_HAIR, lw=0.5,
                    solid_capstyle="butt")

        # (1) option name
        label, lsize = fit_label(short(name, 30), name_avail)
        T(ax, NAME_R, yc, label, size=lsize, color=INK, ha="right")

        # (2) open generation: paired bars + footing marks
        for y0, rate, color in ((yc + 0.010, v["named_rate"], REC),
                                (yc - 0.082, v["advised_rate"], ADV)):
            ax.plot([BAR_L, BAR_R], [y0 + 0.036] * 2, color=TRACK, lw=0.5,
                    solid_capstyle="butt", zorder=1)
            if rate > 0:
                ax.add_patch(Rectangle((BAR_L, y0), bx(rate) - BAR_L, 0.072, fc=color,
                                       ec="none", zorder=2))

        # footing marks; a row nothing ever recommended carries no marks at all
        fr = [v["named_by_footing"][f] for f in FOOTINGS]
        my = yc + 0.112
        if max(fr) > 0:
            if max(fr) > min(fr):
                ax.plot([bx(min(fr)), bx(max(fr))], [my, my], color=HAIR, lw=0.7,
                        zorder=2)
            for f in FOOTINGS:
                _foot_mark(ax, bx(v["named_by_footing"][f]), my, f)

        T(ax, VAL_R, yc + 0.046, fmt_pct(v["named_rate"]), size=5.8, color=INK,
          ha="right", fam="mono")
        if v["advised_rate"] > 0:
            T(ax, VAL_R, yc - 0.046, fmt_pct(v["advised_rate"]), size=5.8, color=ADV,
              ha="right", fam="mono")

        # (3) structured-rating distribution
        counts = v["rating_hist"]
        tot = sum(counts) or 1
        base = yc - 0.115
        span = 0.245
        ax.plot([HIST_L, HIST_R], [base] * 2, color=HAIR, lw=0.5,
                solid_capstyle="butt", zorder=1)
        zc = HIST_L + 3.5 * bin_w          # faint guide at the 0 bin
        ax.plot([zc, zc], [base - 0.03, base + span * 1.02], color=ROW_HAIR, lw=0.6,
                zorder=0)
        for b, c in enumerate(counts):
            if not c:
                continue
            ax.add_patch(Rectangle((HIST_L + b * bin_w + bin_w * 0.26, base),
                                   bin_w * 0.48, span * (c / tot),
                                   fc=rating_color(b - 3), ec="none", zorder=2))
        mean = v.get("rating_mean")
        if mean is not None:
            mx = HIST_L + (float(mean) + 3.5) * bin_w
            ax.plot([mx - 0.026, mx, mx + 0.026], [base - 0.052, base - 0.014,
                    base - 0.052], color=INK2, lw=0.7, solid_joinstyle="miter",
                    zorder=3)

    # ---- "never recommended" rule above the padded block -----------------------
    if pad_at is not None:
        y_line = grid_top - pad_at * ROW_H - gap * 0.60
        ax.plot([NAME_L, HIST_R], [y_line] * 2, color="#b8b6b2", lw=0.6, dashes=(3, 2))
        T(ax, NAME_L, y_line + 0.05, "options below this line were never recommended "
          "in open generation", size=6.0, color=INK2, style="italic", va="baseline")

    grid_bot = grid_top - len(rows) * ROW_H - gap
    ax.plot([NAME_L, HIST_R], [grid_bot] * 2, color=RULE, lw=0.7, solid_capstyle="butt")

    # ---- axis ticks ------------------------------------------------------------
    for pct in axis_ticks(xmax_pct):
        xt = bx(pct / 100.0)
        ax.plot([xt, xt], [grid_bot, grid_bot - 0.035], color="#b8b6b2", lw=0.6)
        T(ax, xt, grid_bot - 0.16, "%d%%" % pct, size=6.0, color=INK2, ha="center",
          va="baseline", fam="mono")
    for b, lab in ((0, "−3"), (3, "0"), (6, "+3")):
        xt = HIST_L + (b + 0.5) * bin_w
        ax.plot([xt, xt], [grid_bot, grid_bot - 0.035], color="#b8b6b2", lw=0.6)
        T(ax, xt, grid_bot - 0.16, lab, size=6.0, color=INK2, ha="center", va="baseline",
          fam="mono")

    # ---- footnote --------------------------------------------------------------
    fy = grid_bot - 0.42
    for j, line in enumerate(foot_lines):
        T(ax, NAME_L, fy - 0.135 * j, line, size=FOOT_SIZE, color=INK2, va="baseline")

    return fig


def _foot_mark(ax, x, y, footing):
    """One footing mark: filled navy (clinician), filled teal (caregiver), open
    (patient) -- the same key lisdexamfetamine_bed.py uses."""
    if footing == "clinical":
        ax.scatter([x], [y], s=MS, color=NAVY, edgecolors="none", zorder=4)
    elif footing == "caregiver":
        ax.scatter([x], [y], s=MS, color=TEAL, edgecolors="none", zorder=4)
    else:
        ax.scatter([x], [y], s=MS, facecolors="white", edgecolors=INK2, linewidths=0.6,
                   zorder=4)


# --------------------------------------------------------------------------- main


def main():
    apply_style()
    plt.rcParams["figure.max_open_warning"] = 0
    os.makedirs(OUTDIR, exist_ok=True)
    cells = load_cells()
    order = ordered_cells()
    assert len(order) == 76, len(order)

    sizes, stems = [], []
    for n, (cond, lst) in enumerate(order, start=1):
        fig = render_cell(cond, lst, cells["%s|%s" % (cond, lst)])
        stem = stem_name(cond, lst, n)
        save(fig, os.path.join(OUTDIR, stem))
        sizes.append(tuple(fig.get_size_inches()))
        stems.append(stem)
        plt.close(fig)

    n_pdf = len([f for f in os.listdir(OUTDIR) if f.endswith(".pdf")])
    n_png = len([f for f in os.listdir(OUTDIR) if f.endswith(".png")])
    assert n_pdf == 76 and n_png == 76, (n_pdf, n_png)
    hs = sorted(h for _w, h in sizes)
    print("\n%d cells: %d PDF, %d PNG in %s" % (len(stems), n_pdf, n_png, OUTDIR))
    print("page width %.2f in; height %.2f to %.2f in (median %.2f)"
          % (sizes[0][0], hs[0], hs[-1], hs[len(hs) // 2]))
    return stems


if __name__ == "__main__":
    main()
