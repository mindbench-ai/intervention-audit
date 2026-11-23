"""Advice disclosures per model: consult-a-professional and
not-a-doctor rates by footing, 41 models grouped by developer.
Data: data/disclosures.json (model_footing keys)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from figstyle import F, INK, INK2, T
from disclosures import AMBER, CM_AMBER, CM_NAVY, FAMS, FOOT
from figstyle import apply_style, save

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "figures", "out")
DATA = os.path.join(ROOT, "data")

# geometry (inches; the axes is aspect-equal, so 1 data unit = 1 inch)
W = 4.9
CW, CH, GAP = 0.505, 0.155, 0.10  # tile width, row pitch, gap between developer blocks
PAD = 0.012                       # inset of the tile inside its row pitch
X_A = 1.55                        # left edge of the consult block
BLOCK_GAP = 0.18                  # gap between the two blocks

# Type sizing. Every explicit point size passes through ts(); the default
# (scale 1.0, no floor) reproduces the authored sizes exactly.
TYPE_SCALE = 1.0
TYPE_MIN = 0.0


def ts(size):
    return max(size * TYPE_SCALE, TYPE_MIN)


def main():
    d = json.load(open(os.path.join(DATA, "disclosures.json")))
    fam_of = {m: v["family"] for m, v in
              json.load(open(os.path.join(ROOT, "config", "models.json")))["models"].items()}
    models = sorted(fam_of)
    # release order within developer (oldest first), by version; the roster records
    # no release dates, so version number is the proxy
    ORDER = ["claude-sonnet-4.5", "claude-haiku-4.5", "claude-opus-4.5", "claude-opus-4.6",
             "claude-sonnet-4.6", "claude-opus-4.8", "claude-sonnet-5", "claude-opus-5",
             "claude-fable-5",
             "gpt-3.5-turbo", "gpt-4.1", "gpt-5-2025-snapshot", "gpt-5-mini-2025-snapshot",
             "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
             "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash",
             "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash",
             "grok-4.3", "grok-4.5", "grok-4.6",
             "mistral-large-3", "ministral-3-8b", "ministral-3-14b", "mistral-small-4",
             "mistral-medium-3.5",
             "qwen3-8b", "qwen-flash", "qwen-3.5-plus", "qwen-3.7-plus", "qwen-3.8-max",
             "deepseek-v4-flash", "deepseek-v4-pro",
             "kimi-k2.6", "kimi-k3"]
    missing = set(models) - set(ORDER)
    assert not missing, missing
    by_fam = {f: [m for m in ORDER if fam_of.get(m) == f] for f in FAMS}

    cw, ch, gap = CW, CH, GAP
    n_rows = len(models) + gap / ch * len(FAMS)
    # 0.52 = bottom margin + the header band below `top`; no footer line is drawn, so
    # nothing but a thin margin sits under the last row.
    H = 0.52 + n_rows * ch + 0.25
    apply_style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    x_a = X_A                          # consult block
    x_b = x_a + 3 * cw + BLOCK_GAP     # not-a-doctor block
    top = H - 0.25
    T(ax, x_a + 1.5 * cw, top, "consult a professional (%)", size=ts(7.2), ha="center",
      fam="serif", weight=600)
    T(ax, x_b + 1.5 * cw, top, "not a doctor (%)", size=ts(7.2), ha="center", fam="serif",
      weight=600)
    for x0 in (x_a, x_b):
        for j, (fk, fn) in enumerate(FOOT):
            T(ax, x0 + j * cw + cw / 2, top - 0.22, fn, size=ts(5.8), color=INK2, ha="center")

    y = top - 0.40
    for f in FAMS:
        y -= gap
        T(ax, 0.12, y - ch / 2, f, size=ts(6.4), color=INK2, fam="sans")
        for m in by_fam[f]:
            y -= ch
            T(ax, x_a - 0.06, y + ch / 2, m, size=ts(5.8), color=INK, ha="right", fam="mono")
            for x0, key, cmap, vmin, vmax in ((x_a, "cp", CM_NAVY, 0.40, 1.0),
                                              (x_b, "nad", CM_AMBER, 0.0, 0.30)):
                for j, (fk, fn) in enumerate(FOOT):
                    v = d[f"model_footing|{m}|{fk}"][key]
                    t = float(min(max((v - vmin) / (vmax - vmin), 0.0), 1.0))
                    xx = x0 + j * cw
                    ax.add_patch(Rectangle((xx + PAD, y + PAD), cw - 2 * PAD, ch - 2 * PAD,
                                           fc=cmap(t), ec="none", zorder=2))
                    T(ax, xx + cw / 2, y + ch / 2, f"{100 * v:.0f}", size=ts(5.6),
                      color="white" if t > 0.55 else INK, ha="center", fam="mono")

    os.makedirs(OUT, exist_ok=True)
    save(fig, os.path.join(OUT, "disclosures_by_model"))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
