"""Model behavior: breadth against naming panel-rejected options.

One point per model (41). x = recommended interventions per response. y = share of the
model's list-mapped recommendations whose panel mean rating is below zero, with the model's
own ratings left out of the panel where it was rated. Top panel: over each cell's two asks.
Bottom panel: the open ask only. Colour = developer; filled = also a rating model, hollow =
generation only. Data: data/models.json.

Labels are placed by a greedy candidate-offset search (right, left, diagonals, above,
below at growing radii). A label that cannot find a clean slot falls back to a number
and is listed in the key drawn in the empty corner of the top panel.
"""

import json
import math
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
FAMS = ["Anthropic", "OpenAI", "Google", "xAI", "Mistral", "Qwen", "DeepSeek", "Moonshot"]
PANELS = [("share_rated_below_zero", "rec_per_response", "both asks in the cell", 23),
          ("share_rated_below_zero_open", "rec_per_response_open", "open ask only", 11.5)]
GRID = "#e6e6e6"

LBL_SIZE = 5.4          # model-name labels (mono)
KEY_SIZE = 5.4          # key entries (mono)
MARK_S = 20             # marker area, points^2
MARK_R = 2.9            # marker radius, points (s=20 -> r = 2.5pt, plus the edge)
# candidate slots: stand-off distances (points) x placement rules. The rule says where the
# label box sits relative to the point: R/L/C(+/-) horizontally, U/D/M vertically.
CAND_D = [2.0, 3.5, 5.5, 8.0, 11.0, 15.0, 20.0]
CAND_P = [("R", "M"), ("L", "M"), ("R", "U"), ("R", "D"), ("L", "U"), ("L", "D"),
          ("C", "U"), ("C", "D"), ("C+", "U"), ("C-", "U"), ("C+", "D"), ("C-", "D")]
LEAD_MIN = 4.5          # draw a leader line once the stand-off exceeds this, points


def short_name(m):
    """Shorten a model key for its label: drop the claude- and gemini- prefixes (the developer
    is carried by color) and replace the hyphen after the other developer prefixes with a space."""
    for pre in ("claude-", "gemini-", "gpt-", "grok-", "qwen-", "deepseek-", "kimi-", "mistral-", "ministral-"):
        if m.startswith(pre):
            return {"gpt-": "gpt ", "grok-": "grok ", "kimi-": "kimi ", "mistral-": "mistral ", "ministral-": "ministral ",
                    "deepseek-": "deepseek ", "qwen-": "qwen "}.get(pre, "") + m[len(pre):]
    return m


# ---------------------------------------------------------------- geometry helpers

def _hit(a, b, pad=0.0):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def _inside(box, lim):
    return box[0] >= lim[0] and box[1] >= lim[1] and box[2] <= lim[2] and box[3] <= lim[3]


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    ll = dx * dx + dy * dy
    t = 0.0 if ll <= 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / ll))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _seg_box(a, b, box):
    """Liang-Barsky: does the segment a->b cut the rectangle box?"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - box[0]), (dx, box[2] - a[0]),
                 (-dy, a[1] - box[1]), (dy, box[3] - a[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return True


def _candidate(pt, rule, d, w, h, rm):
    """Box (x0, y0, x1, y1) for a label of size w x h in slot `rule` at stand-off d."""
    px, py = pt
    hx, hy = rule
    x0 = {"R": px + rm + d, "L": px - rm - d - w, "C": px - w / 2.0,
          "C+": px - 0.15 * w, "C-": px - 0.85 * w}[hx]
    y0 = {"U": py + rm + 0.55 * d, "D": py - rm - 0.55 * d - h, "M": py - h / 2.0}[hy]
    return (x0, y0, x0 + w, y0 + h)


def _anchor(box, pt):
    """Point on the label box closest to pt, used as the foot of the leader line."""
    return (min(max(pt[0], box[0]), box[2]), min(max(pt[1], box[1]), box[3]))


# ---------------------------------------------------------------- label placement

def place_labels(ax, items, reserved, fontsize):
    """Greedy placement. items = [(x, y, text)]. Returns (indices that found no clean
    slot, artists created). Every label is drawn; a failed one takes its least-bad slot."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    sc = fig.dpi / 72.0
    rm = MARK_R * sc
    pts = [tuple(ax.transData.transform((x, y))) for x, y, _ in items]
    texts, size = [], []
    for (x, y, s), p in zip(items, pts):
        t = ax.text(x, y, s, fontsize=fontsize, color=INK, fontfamily=F["mono"],
                    ha="left", va="center", zorder=4)
        bb = t.get_window_extent(rend)
        texts.append(t)
        size.append((bb.width, bb.height))
    dot_boxes = [(p[0] - rm, p[1] - rm, p[0] + rm, p[1] + rm) for p in pts]
    axbb = ax.get_window_extent(rend)
    lim = (axbb.x0 + 1.5 * sc, axbb.y0 + 1.5 * sc, axbb.x1 - 1.5 * sc, axbb.y1 - 1.5 * sc)
    static = list(reserved) + dot_boxes

    def crowding(i):
        return sum(1 for j, q in enumerate(pts)
                   if j != i and math.hypot(q[0] - pts[i][0], q[1] - pts[i][1]) < 42 * sc)

    order = sorted(range(len(items)), key=lambda i: (-crowding(i), pts[i][0]))
    taken, failed, arts = [], [], []
    for i in order:
        w, h = size[i]
        best, best_cost, best_d = None, None, 0.0
        for d in CAND_D:
            for u in CAND_P:
                box = _candidate(pts[i], u, d * sc, w, h, rm)
                if not _inside(box, lim):
                    continue
                cost = 0.0
                for ob in static:
                    if _hit(box, ob, 0.8 * sc):
                        cost += 1.0
                for ob in taken:
                    if _hit(box, ob, 0.8 * sc):
                        cost += 1.0
                foot = _anchor(box, pts[i])
                if d > LEAD_MIN:      # a leader must miss other points, labels and the key
                    for j, q in enumerate(pts):
                        if j == i or math.dist(q, pts[i]) < 1.6 * rm:
                            continue   # a twin sitting on this point cannot be avoided
                        if _seg_dist(q, pts[i], foot) < rm + 0.8 * sc:
                            cost += 1.0
                    for ob in taken + list(reserved):
                        if _seg_box(pts[i], foot, ob):
                            cost += 1.0
                if cost == 0:
                    best, best_cost, best_d = box, 0.0, d
                    break
                if best_cost is None or cost < best_cost:
                    best, best_cost, best_d = box, cost, d
            if best_cost == 0:
                break
        if best is None:   # nothing fitted inside the axes at all
            best, best_cost, best_d = _candidate(pts[i], ("R", "M"), 2.0 * sc, w, h, rm), 1.0, 2.0
        if best_cost > 0:
            failed.append(i)
            if os.environ.get("FIGDEBUG"):
                why = []
                for k, ob in enumerate(static):
                    if _hit(best, ob, 0.8 * sc):
                        why.append("res" if k < len(reserved) else f"dot{items[k - len(reserved)][2]}")
                for k, ob in enumerate(taken):
                    if _hit(best, ob, 0.8 * sc):
                        why.append("lbl")
                print(f"    ! {items[i][2]!r} cost={best_cost} d={best_d} {why}")
        taken.append(best)
        x0, yc = best[0], (best[1] + best[3]) / 2.0
        texts[i].set_position(ax.transData.inverted().transform((x0, yc)))
        arts.append(texts[i])
        if best_d > LEAD_MIN:
            foot = _anchor(best, pts[i])
            fx, fy = ax.transData.inverted().transform(foot)
            arts.append(ax.annotate("", xy=(items[i][0], items[i][1]), xytext=(fx, fy),
                                    textcoords="data", zorder=2,
                                    arrowprops=dict(arrowstyle="-", color=HAIR, lw=0.5,
                                                    shrinkA=1.0, shrinkB=2.6)))
    return set(failed), arts


# ---------------------------------------------------------------- numbered key

def key_layout(ax, entries, fontsize):
    """Lay the key out in the empty lower-right of the panel; return (box, positions)."""
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    sc = fig.dpi / 72.0
    axbb = ax.get_window_extent(rend)
    left = axbb.x0 + 0.545 * axbb.width
    top = axbb.y0 + 0.60 * axbb.height
    bottom = axbb.y0 + 2 * sc
    probe = ax.text(0, 0, max(entries, key=len), fontsize=fontsize, fontfamily=F["mono"])
    pb = probe.get_window_extent(rend)
    probe.remove()
    lh = 1.62 * fontsize * sc
    rows = max(1, int((top - bottom - 1.9 * lh) // lh))
    ncol = max(1, math.ceil(len(entries) / rows))
    rows = math.ceil(len(entries) / ncol)
    colw = pb.width + 7 * sc
    pos, x = [], left
    for c in range(ncol):
        for r in range(rows):
            k = c * rows + r
            if k < len(entries):
                pos.append((x, top - 1.55 * lh - r * lh))
        x += colw
    box = (left - 4 * sc, top - 1.55 * lh - (rows - 1) * lh - 0.7 * lh,
           left + ncol * colw, top + 0.6 * lh)
    return box, pos, top


def draw_key(ax, entries, fontsize):
    """Draw the key and return (reserved display box, artists)."""
    box, pos, top = key_layout(ax, entries, fontsize)
    inv = ax.transData.inverted()
    arts = []
    hx, hy = inv.transform((box[0] + 4 * (ax.figure.dpi / 72.0), top))
    arts.append(ax.text(hx, hy, "numbered points", fontsize=6.0, color=INK2,
                        fontfamily=F["mono"], ha="left", va="center", zorder=5))
    for (x, y), s in zip(pos, entries):
        dx, dy = inv.transform((x, y))
        arts.append(ax.text(dx, dy, s, fontsize=fontsize, color=INK, fontfamily=F["mono"],
                            ha="left", va="center", zorder=5))
    return box, arts


# ---------------------------------------------------------------- figure

def main():
    apply_style()
    models = json.load(open(os.path.join(ROOT, "data", "models.json")))["models"]
    order = [m for m, _ in sorted(models.items(), key=lambda kv: kv[1]["rec_per_response"])]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.9), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.955, bottom=0.105, hspace=0.20)

    for ax, (ykey, xkey, sub, ymax) in zip(axes, PANELS):
        for m in order:
            v = models[m]
            c = FAM_COLOR[v["family"]]
            ax.scatter([v[xkey]], [100 * v[ykey]], s=MARK_S, color=c if v["rated"] else "white",
                       edgecolors=c, linewidths=0.9, zorder=3)
        ax.set_xlim(0, 32)
        ax.set_ylim(-0.4, ymax)
        ax.set_yticks([0, 5, 10, 15, 20] if ymax > 15 else [0, 5, 10])
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.tick_params(labelsize=7.2, length=2.5)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(F["mono"])
            lbl.set_color(INK2)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(HAIR)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(sub, fontsize=7.2, fontfamily=F["mono"], color=INK2, loc="left", pad=4)
        ax.set_ylabel("% of its recommendations\nrated below 0 by the other models",
                      fontsize=7.2, color=INK2, fontfamily=F["sans"])
    axes[1].set_xlabel("recommended interventions per response", fontsize=7.2, color=INK2,
                       fontfamily=F["sans"])

    # legend: developers, plus the filled / hollow key
    ax = axes[0]
    leg = []
    x0, y0 = 0.03, 0.97
    for i, f in enumerate(FAMS):
        col, row = i // 4, i % 4
        leg.append(ax.scatter([x0 + col * 0.19], [y0 - row * 0.062], s=22, color=FAM_COLOR[f],
                              transform=ax.transAxes, zorder=5, clip_on=False))
        leg.append(ax.text(x0 + col * 0.19 + 0.025, y0 - row * 0.062, f, transform=ax.transAxes,
                           fontsize=6.4, color=INK, fontfamily=F["sans"], va="center"))
    yk = y0 - 4.35 * 0.062
    leg.append(ax.scatter([x0], [yk], s=22, color=INK2, transform=ax.transAxes, zorder=5))
    leg.append(ax.text(x0 + 0.025, yk, "also a rating model", transform=ax.transAxes,
                       fontsize=6.0, color=INK2, fontfamily=F["sans"], va="center"))
    leg.append(ax.scatter([x0 + 0.19], [yk], s=22, color="white", edgecolors=INK2,
                          linewidths=0.9, transform=ax.transAxes, zorder=5))
    leg.append(ax.text(x0 + 0.19 + 0.025, yk, "generation only", transform=ax.transAxes,
                       fontsize=6.0, color=INK2, fontfamily=F["sans"], va="center"))

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    lb = [a.get_window_extent(rend) for a in leg]
    legend_box = (min(b.x0 for b in lb) - 3, min(b.y0 for b in lb) - 3,
                  max(b.x1 for b in lb) + 3, max(b.y1 for b in lb) + 3)

    # how tightly each model sits among the others, in both panels at once
    crowd = {m: 0.0 for m in order}
    nrm2 = []
    for ykey, xkey, _s, ymax in PANELS:
        nrm = {m: (models[m][xkey] / 32.0, 100 * models[m][ykey] / (ymax + 0.4)) for m in order}
        nrm2.append(nrm)
        for m in order:
            for n in order:
                if n != m and math.dist(nrm[m], nrm[n]) < 0.075:
                    crowd[m] += 1.0

    # iterate: labels that cannot be placed cleanly become numbers listed in the key
    keyed, arts, entries = set(), [], []
    for _ in range(14):
        for a in arts:
            a.remove()
        arts = []
        num = {m: str(i + 1) for i, m in enumerate([m for m in order if m in keyed])}
        entries = [f"{num[m]:>2s}  {short_name(m)}" for m in order if m in keyed]
        key_box = None
        if entries:
            key_box, keyarts = draw_key(axes[0], entries, KEY_SIZE)
            arts += keyarts
        failed = set()
        for ax_, (ykey, xkey, sub, ymax) in zip(axes, PANELS):
            items = [(models[m][xkey], 100 * models[m][ykey], num.get(m, short_name(m)))
                     for m in order]
            res = [legend_box] + ([key_box] if key_box else []) if ax_ is axes[0] else []
            bad, a2 = place_labels(ax_, items, res, LBL_SIZE)
            arts += a2
            failed |= {order[i] for i in bad}
        if not failed:
            print(f"placed clean with {len(keyed)} keyed")
            break
        new = failed - keyed
        if not new:   # the stragglers are already numbers: free space around them
            free = [m for m in order if m not in keyed]
            new = set()
            for m in failed:
                near = sorted(free, key=lambda n: min(
                    math.dist(nrm2[k][m], nrm2[k][n]) for k in range(2)))
                if near:
                    new.add(near[0])
        print(f"pass: {len(keyed)} keyed, {len(failed)} unplaced, keying {sorted(new)}")
        keyed |= new

    fig.text(0.09, 0.030, "One point per model, 41 models. Panel rating excludes the model's own ratings "
             "where it was rated.", fontsize=6.4, color=INK2, fontfamily=F["sans"])
    fig.text(0.09, 0.011, "Recommendations counted over each cell's two asks (top panel) or the open ask "
             "only (bottom panel).", fontsize=6.4, color=INK2, fontfamily=F["sans"])

    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "model_breadth"))

    # audit table: every model with its x and y in both panels, straight from models.json
    print("\n%-26s %6s %6s   %6s %6s" % ("model", "x_top", "y_top", "x_bot", "y_bot"))
    for m in order:
        v = models[m]
        print("%-26s %6.2f %6.2f   %6.2f %6.2f" % (
            m, v["rec_per_response"], 100 * v["share_rated_below_zero"],
            v["rec_per_response_open"], 100 * v["share_rated_below_zero_open"]))


if __name__ == "__main__":
    main()
