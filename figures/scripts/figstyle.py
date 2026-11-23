"""Shared matplotlib style for the intervention-audit figure suite."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts")

# project palette and type, shared by every figure
INK = "#2d2c2b"
INK2 = (0, 0, 0, 0.58)
HAIR = (0, 0, 0, 0.12)
PANEL = "#F5F5F4"
NAVY = "#1B3A5C"
NAVY_L = "#8FA7BD"
TEAL = "#4A8DB0"
TEAL_L = "#A9C9DB"


def register_fonts():
    """Register the bundled fonts (figures/fonts/) and return the family name for each role.
    Source Serif 4 is used for numerals and titles, Instrument Sans for labels, and
    JetBrains Mono for stamps. Missing files fall back to the DejaVu families."""
    fams = {}
    for stem, key in (("instrument-sans", "sans"), ("source-serif-4", "serif"),
                      ("jetbrains-mono", "mono")):
        p = os.path.join(FONTS, f"{stem}.ttf")
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            fams[key] = font_manager.FontProperties(fname=p).get_name()
    return {"sans": fams.get("sans", "DejaVu Sans"),
            "serif": fams.get("serif", "DejaVu Serif"),
            "mono": fams.get("mono", "DejaVu Sans Mono")}


F = register_fonts()


def T(ax, x, y, s, size=7, color=INK, ha="left", va="center", fam="sans", weight="normal"):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontfamily=F[fam],
                   fontweight=weight)


# palette
GREEN = "#1baf7a"   # recommended / positive
ORANGE = "#eb6834"  # advised-against / negative
GRAY = "#8a8880"    # neutral
PURPLE = "#4a3aa7"  # mixed
BLUE = "#2a78d6"    # accent / highlight
MUTED = "#52514e"   # muted text
GRID = "#cccccc"

FAMILY_COLORS = {
    "Anthropic": BLUE,
    "OpenAI": ORANGE,
    "Google": GREEN,
    "xAI": "#eda100",
    "Mistral": "#e87ba4",
    "Qwen": PURPLE,
    "DeepSeek": MUTED,
    "Moonshot": GRAY,
}


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, stem):
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=200)
    print(f"wrote {stem}.pdf / {stem}.png")
