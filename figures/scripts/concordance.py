"""Concordance between the two arms, by ask.

Every (cell, option) pair is a point: x = mean rating in the structured arm, y = share of
the cell's open-generation responses recommending the option. One column per
intervention list. Top row: recommendation rate on the list's own ask (medication / therapy /
procedures / apps). Bottom row: recommendation rate on the open ask. Options recommended in at least 5%
of responses yet rated below zero are amber; the survivors on the open ask are labelled.
With --numbered, the five amber points with the highest named rate in each panel (ties to
the lower rating) are numbered instead and listed in a corner key, and the numbered points
are printed as a verification table. Data: data/cells.json (named_by_ask, rating_mean).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from cellutil import COND_DISPLAY, LIST_LABEL, LISTS, OUT, load_cells, short  # noqa: E402
from figstyle import F, HAIR, INK, INK2, NAVY, T  # noqa: E402
from figstyle import apply_style, save  # noqa: E402

AMBER = "#D9822B"
THRESH = 5.0
NUMBERED = "--numbered" in sys.argv   # alternative view: number the amber points, key in a corner
NKEY = 5                              # amber points numbered per panel (highest named rate first)
KEY_COND = {"Bipolar I (maint.)": "Bipolar I", "Schizophrenia (1st ep.)": "Schizophrenia",
            "Binge-eating disorder": "Binge eating", "Alcohol use disorder": "Alcohol use"}
LIST_ASK = {"pharmacological": "medication", "psychotherapy": "psychotherapy",
            "neuromodulation": "neuromodulation", "digital": "digital"}
ASK_LABEL = {"medication": "medication ask", "psychotherapy": "therapy ask",
             "neuromodulation": "procedures ask", "digital": "apps ask"}
# hand-placed labels: (row, list, option, condition) -> (text, dx, dy, ha)
# hand offsets for the one label per panel: (row, list) -> (dx, dy, ha); default is right of the dot
LABEL_POS = {
    ("type", "pharmacological"): (0, 11, "center"),
    ("type", "psychotherapy"): (0, 11, "center"),
    ("type", "neuromodulation"): (6, 0, "left"),
    ("type", "digital"): (0, 11, "center"),
    ("open", "pharmacological"): (-6, 0, "right"),
    ("open", "psychotherapy"): (0, 11, "center"),
    ("open", "neuromodulation"): (6, 0, "left"),
}
COND_SHORT = {"Bipolar I (maint.)": "bipolar I", "Schizophrenia (1st ep.)": "schizophrenia", "Binge-eating disorder": "binge eating",
              "Alcohol use disorder": "alcohol use", "Autism spectrum": "autism", "Borderline PD": "borderline PD",
              "Bereavement": "bereavement"}


XLIM, YLIM = (-3.15, 3.15), (-3, 103)
DOT_S = 40.0        # area of a numbered dot, pt^2 (diameter ~6.3 pt)
DIGIT_PT = 5.0      # digit size on the dot
MERGE_PT = 5.0      # numbered dots closer than this share one combined digit
SIDE_PT = 5.4       # that combined digit, set beside the blob
KEY_PT = 5.7
KEY_LEAD = 1.35


def pt_per_unit(fig, ax):
    """Points per data unit along x and y, from the fixed axes box and limits."""
    pos = ax.get_position()
    w, h = fig.get_size_inches()
    return (72 * w * pos.width / (XLIM[1] - XLIM[0]),
            72 * h * pos.height / (YLIM[1] - YLIM[0]))


def dot_groups(worst, sx, sy):
    """Group the numbered points that sit within MERGE_PT of one another.

    Every dot is still drawn at its exact (x, y); only the digits are merged, so a
    pair that renders as a single blob carries one combined label (e.g. "1·2").
    """
    groups = []
    for i, (x, y, _, _) in enumerate(worst, 1):
        for g in groups:
            if any(((x - qx) * sx) ** 2 + ((y - qy) * sy) ** 2 < MERGE_PT ** 2 for qx, qy, _ in g):
                g.append((x, y, i))
                break
        else:
            groups.append([(x, y, i)])
    return groups


def side_label(ax, g, lab, pts, sx, sy):
    """Combined digit for dots too close to carry one each: set beside the blob, on a tick.

    The dots stay exactly where the data puts them; only the digits move out.
    """
    gx = sum(p[0] for p in g) / len(g)
    gy = sum(p[1] for p in g) / len(g)
    hw = max(abs(p[0] - gx) * sx for p in g) + DOT_S ** 0.5 / 2      # blob half-width, pt
    wpt = SIDE_PT * 0.62 * len(lab)
    d, room = 1, None
    for side in (1, -1):
        a = gx + side * (hw + 2.5) / sx
        b = a + side * wpt / sx
        n = sum(1 for x, y, _, _ in pts
                if min(a, b) <= x <= max(a, b) and abs(y - gy) * sy <= 4.5)
        if room is None or n < room:
            d, room = side, n
    ax.plot([gx + d * (hw + 0.4) / sx, gx + d * (hw + 2.2) / sx], [gy, gy], color=AMBER, lw=0.6,
            solid_capstyle="butt", zorder=6)
    ax.text(gx + d * (hw + 2.8) / sx, gy, lab, ha="left" if d > 0 else "right", va="center",
            fontsize=SIDE_PT, color=AMBER, fontfamily=F["sans"], fontweight="bold", zorder=6,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def key_rows(lines, side, sx, sy):
    """Per-line boxes of the key, in data units: (x0, x1, y_centre, y0, y1)."""
    lead = KEY_PT * KEY_LEAD
    top = YLIM[1] - 0.03 * (YLIM[1] - YLIM[0])
    rows = []
    for i, s in enumerate(lines):
        yc = top - (i + 0.5) * lead / sy
        w = KEY_PT * 0.56 * len(s) / sx
        if side == "right":
            x1 = XLIM[1] - 0.02 * (XLIM[1] - XLIM[0]); x0 = x1 - w
        else:
            x0 = XLIM[0] + 0.02 * (XLIM[1] - XLIM[0]); x1 = x0 + w
        rows.append((x0, x1, yc, yc - 0.5 * lead / sy, yc + 0.5 * lead / sy))
    return rows


def key_corner(pts, lines, sx, sy):
    """Top corner for the key: never over an amber point, then over the fewest grey ones.

    Scored line by line, so a ragged key can sit beside a point that a full-width box
    would have claimed to cover.
    """
    score = {}
    for side in ("left", "right"):
        amber = grey = 0
        rows = key_rows(lines, side, sx, sy)
        for x, y, _, _ in pts:
            if any(x0 - 2 / sx <= x <= x1 + 2 / sx and y0 - 2 / sy <= y <= y1 + 2 / sy
                   for x0, x1, _, y0, y1 in rows):
                if y >= THRESH and x < 0:
                    amber += 1
                else:
                    grey += 1
        score[side] = (amber > 0, grey)
    return "right" if score["right"] < score["left"] else "left"


def points(cells, lst, view):
    ask = "open" if view == "open" else LIST_ASK[lst]
    pts = []
    for key, cell in cells.items():
        if key.startswith("_") or not key.endswith("|" + lst):
            continue
        cond = key.split("|")[0]
        for nm, o in cell["options"].items():
            if o["is_null"] or o["rating_mean"] is None:
                continue
            pts.append((o["rating_mean"], 100 * o["named_by_ask"][ask], nm, cond))
    return pts


def main():
    apply_style()
    cells = load_cells()
    # No in-figure footer, so the bottom margin only has to clear the x tick labels
    # and the axis title.
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 5.30), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.894, bottom=0.079, wspace=0.12, hspace=0.40)
    n_total = 0
    table, keys = [], []
    for row, view in enumerate(("type", "open")):
        for col, lst in enumerate(LISTS):
            ax = axes[row][col]
            ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
            pts = points(cells, lst, view)
            n_total += len(pts) if row == 0 else 0
            base = [(x, y) for x, y, _, _ in pts if not (y >= THRESH and x < 0)]
            flag = [(x, y, nm, c) for x, y, nm, c in pts if y >= THRESH and x < 0]
            ax.scatter([x for x, _ in base], [y for _, y in base], s=7, color=NAVY, alpha=0.35,
                       linewidths=0, zorder=2)
            ax.scatter([x for x, _, _, _ in flag], [y for _, y, _, _ in flag], s=16, color=AMBER,
                       edgecolors="white", linewidths=0.4, zorder=3)
            if NUMBERED:
                # worst five: highest named rate among the amber points, ties to the lower rating
                worst = sorted(flag, key=lambda t: (-t[1], t[0]))[:NKEY]
                table.append((view, lst, worst))
                sx, sy = pt_per_unit(fig, ax)
                # every numbered dot is drawn at its exact (x, y); nothing is nudged
                ax.scatter([x for x, _, _, _ in worst], [y for _, y, _, _ in worst], s=DOT_S,
                           color=AMBER, edgecolors="white", linewidths=0.5, zorder=5)
                for g in dot_groups(worst, sx, sy):
                    lab = "·".join(str(i) for _, _, i in sorted(g, key=lambda t: t[2]))
                    if len(g) == 1:
                        ax.text(g[0][0], g[0][1], lab, ha="center", va="center", fontsize=DIGIT_PT,
                                color="white", fontfamily=F["sans"], fontweight="bold", zorder=6)
                    else:
                        side_label(ax, g, lab, pts, sx, sy)
                if worst:
                    lines = ["%d  %s · %s" % (i, short(nm, 18), KEY_COND.get(COND_DISPLAY[c], COND_DISPLAY[c]))
                             for i, (_, _, nm, c) in enumerate(worst, 1)]
                    side = key_corner(pts, lines, sx, sy)
                    rows = key_rows(lines, side, sx, sy)
                    for s, (x0, x1, yc, _, _) in zip(lines, rows):
                        t = ax.text(x1 if side == "right" else x0, yc, s,
                                    ha="right" if side == "right" else "left", va="center",
                                    fontsize=KEY_PT, color=INK, fontfamily=F["sans"], zorder=7,
                                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
                        keys.append((ax, t, flag, view, lst))
            elif flag:
                # label the single worst offender: highest named rate among the amber points
                x, y, nm, c = max(flag, key=lambda t: (t[1], -t[0]))
                dx, dy, ha = LABEL_POS.get((view, lst), (6, 0, "left"))
                cond = COND_DISPLAY[c]; cond = COND_SHORT.get(cond, cond)
                ax.annotate("%s · %s" % (short(nm, 22), cond), (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va="center", fontsize=6.2, color=INK, fontfamily=F["sans"], zorder=4,
                            arrowprops=dict(arrowstyle="-", color=HAIR, lw=0.5, shrinkA=0, shrinkB=2.5) if dy else None)
            n_named = sum(1 for _, y, _, _ in pts if y >= THRESH)
            if row == 0:
                ax.set_title(LIST_LABEL[lst], fontsize=10, fontfamily=F["serif"], fontweight=600,
                             color=INK, loc="left", pad=26)
            sub = ASK_LABEL[LIST_ASK[lst]] if view == "type" else "unconstrained ask"
            ax.text(0.0, 1.10, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8,
                    color=INK2, fontfamily=F["mono"])
            ax.text(0.0, 1.02, f"{len(flag)} of {n_named} rated below 0", transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=6.6, color=AMBER if flag else INK2, fontfamily=F["sans"])
            ax.set_xticks([-3, 0, 3]); ax.set_yticks([0, 50, 100])
            ax.axvline(0, color=HAIR, lw=0.6, zorder=1)
            ax.axhline(THRESH, color=HAIR, lw=0.5, ls=(0, (2, 2)), zorder=1)
            ax.tick_params(labelsize=7.2, length=2.5)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontfamily(F["mono"]); lbl.set_color(INK2)
            ax.spines["left"].set_color(HAIR); ax.spines["bottom"].set_color(HAIR)
            if row == 1:
                ax.set_xlabel("mean rating, structured arm", fontsize=7.2, color=INK2,
                              fontfamily=F["sans"], labelpad=3)
        axes[row][0].set_ylabel("% of responses\nrecommending it", fontsize=7.2, color=INK2,
                                fontfamily=F["sans"])
    print("points per panel row: %s in 76 cells" % format(n_total, ","))
    os.makedirs(OUT, exist_ok=True)
    if NUMBERED:
        check_keys(fig, keys)
    save(fig, os.path.join(OUT, "concordance_numbered" if NUMBERED else "concordance"))
    if NUMBERED:
        report(table)


def check_keys(fig, keys):
    """Measure each key and report any amber point it would cover."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = 2.0 * fig.dpi / 72
    for ax, t, flag, view, lst in keys:
        bb = t.get_window_extent(renderer=r).expanded(1.0, 1.0)
        hits = [(nm, c) for x, y, nm, c in flag
                if bb.x0 - pad <= ax.transData.transform((x, y))[0] <= bb.x1 + pad
                and bb.y0 - pad <= ax.transData.transform((x, y))[1] <= bb.y1 + pad]
        if hits:
            print("KEY OVERLAP %s/%s: %s" % (view, lst, hits))


def report(table):
    """Verification table: the numbered points of every panel, straight from cells.json."""
    print("\npanel                        #  option                          condition"
          "                x (rating_mean)   y (named %)")
    for view, lst, worst in table:
        panel = "%s / %s" % ("own ask" if view == "type" else "open ask", lst)
        for i, (x, y, nm, c) in enumerate(worst, 1):
            print("%-28s %d  %-31s %-22s %9.4f %12.4f" % (panel if i == 1 else "", i, nm,
                                                          COND_DISPLAY[c], x, y))


if __name__ == "__main__":
    main()
