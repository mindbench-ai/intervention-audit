"""Advice disclosures: consult-a-professional and not-a-doctor rates
by footing and ask, with marginals. Data: data/disclosures.json (per-response judge
booleans aggregated over the full generation corpus, n = 35,055).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from figstyle import F, HAIR, INK, INK2, NAVY, PANEL, T
from figstyle import apply_style, save

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "figures", "out")
DATA = os.path.join(ROOT, "data")
AMBER = "#B8845C"

FOOT = [("clinical", "clinician"), ("caregiver", "caregiver"), ("patient", "patient")]
ASKS = [("open", "unconstrained"), ("medication", "medication"), ("psychotherapy", "therapy"),
        ("neuromodulation", "procedures"), ("digital", "digital")]
FAMS = ["Anthropic", "OpenAI", "Google", "xAI", "Mistral", "Qwen", "DeepSeek", "Moonshot"]
COND = {
    "major_depressive_disorder": "major depression", "generalized_anxiety_disorder": "generalized anxiety",
    "panic_disorder": "panic disorder", "ocd": "OCD", "ptsd": "PTSD", "chronic_insomnia": "chronic insomnia",
    "social_anxiety_disorder": "social anxiety", "adhd_adult": "adult ADHD",
    "binge_eating_disorder": "binge eating", "bipolar_1_maintenance": "bipolar I (maintenance)",
    "alcohol_use_disorder": "alcohol use disorder", "schizophrenia_first_episode": "1st-episode schizophrenia",
    "bulimia_nervosa": "bulimia nervosa", "anorexia_nervosa": "anorexia nervosa",
    "borderline_personality_disorder": "borderline PD", "narcissistic_personality_disorder": "narcissistic PD",
    "autism_spectrum_disorder": "autism spectrum", "situational_stress_subclinical": "situational stress *",
    "uncomplicated_bereavement": "bereavement *",
}

CM_NAVY = LinearSegmentedColormap.from_list("navy", ["#FFFFFF", NAVY])
CM_AMBER = LinearSegmentedColormap.from_list("amber", ["#FFFFFF", AMBER])


def grid(ax, x0, y0, cw, ch, rows, cols, val, cmap, vmin, vmax, row_lbl, col_lbl, fmt="{:.0f}",
         col_lbl_top=True, lbl_size=6.2):
    """Draw a value grid with its own labels; (x0,y0) is the top-left of the cell block."""
    for i, (rk, rn) in enumerate(rows):
        yy = y0 - (i + 1) * ch
        T(ax, x0 - 0.06, yy + ch / 2, rn, size=lbl_size, color=INK, ha="right")
        for j, (ck, cn) in enumerate(cols):
            xx = x0 + j * cw
            v = val(rk, ck)
            t = (v - vmin) / (vmax - vmin)
            t = float(min(max(t, 0.0), 1.0))
            ax.add_patch(Rectangle((xx + 0.012, yy + 0.012), cw - 0.024, ch - 0.024,
                                   fc=cmap(t), ec="none", zorder=2))
            T(ax, xx + cw / 2, yy + ch / 2, fmt.format(100 * v), size=6.0,
              color="white" if t > 0.55 else INK, ha="center", fam="mono")
    if col_lbl_top:
        for j, (ck, cn) in enumerate(cols):
            T(ax, x0 + j * cw + cw / 2, y0 + 0.06, cn, size=lbl_size, color=INK2, ha="center",
              va="bottom")


def main():
    d = json.load(open(os.path.join(DATA, "disclosures.json")))

    def g(key, m):
        return d[key][m]

    apply_style()
    # one column, 6.2 in wide; A stacked above B, each grid spanning the width
    W = 6.2
    x0 = 0.15                                  # left edge of each panel
    gx = x0 + 0.70                             # left edge of the cell block
    cw = (W - 0.15 - 0.48 - gx) / 5            # cells fill the width up to the margin column
    ch = 0.30
    PANEL_H = 0.62 + 3 * ch + 0.13             # head center to the bottom-margin row
    STACK_GAP = 0.42                           # bottom-margin row of A to head of B
    H = 0.22 + 2 * PANEL_H + STACK_GAP + 0.16
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    def head(x, y, letter, title, sub=None):
        ax.add_patch(plt.Circle((x, y), 0.085, fc=INK, ec="none", zorder=1))
        T(ax, x, y, letter, size=7, color="white", ha="center")
        T(ax, x + 0.16, y, title, size=8.6, fam="serif", weight=600)
        if sub:
            T(ax, x + 0.16, y - 0.17, sub, size=6.2, color=INK2)

    # A consult above B not a doctor -- same grid, full width each
    for k, (letter, title, key, cmap, vmin, vmax) in enumerate((
            ("A", "Consult a professional", "cp", CM_NAVY, 0.40, 1.0),
            ("B", "Not a doctor", "nad", CM_AMBER, 0.0, 0.30))):
        top = H - 0.22 - k * (PANEL_H + STACK_GAP)
        head(x0 + 0.10, top, letter, title, "% of responses, by footing and ask")
        grid(ax, gx, top - 0.62, cw, ch, FOOT, ASKS,
             lambda f, a, key=key: g(f"footing_ask|{f}|{a}", key), cmap, vmin, vmax,
             None, None, lbl_size=5.8)
        # row margin: all asks
        xx = gx + 5 * cw + 0.08
        T(ax, xx + 0.20, top - 0.62 + 0.06, "all asks", size=5.8, color=INK2, ha="center",
          va="bottom")
        for i, (fk, fn) in enumerate(FOOT):
            yy = top - 0.62 - (i + 1) * ch
            T(ax, xx + 0.20, yy + ch / 2, f"{100 * g(f'footing|{fk}', key):.0f}", size=6.0,
              color=INK2, ha="center", fam="mono")
        # bottom margin: all footings
        yy = top - 0.62 - 3 * ch - 0.04
        T(ax, gx - 0.06, yy - 0.09, "all footings", size=5.8, color=INK2, ha="right")
        for j, (ak, an) in enumerate(ASKS):
            T(ax, gx + j * cw + cw / 2, yy - 0.09, f"{100 * g(f'ask|{ak}', key):.0f}",
              size=6.0, color=INK2, ha="center", fam="mono")
        T(ax, xx + 0.20, yy - 0.09, f"{100 * g('all', key):.0f}", size=6.0, color=INK,
          ha="center", fam="mono")

    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "disclosures"))


if __name__ == "__main__":
    main()
