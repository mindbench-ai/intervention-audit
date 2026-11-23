"""cells.json -> data/concordance.json: how the two arms relate, per cell and overall.

For every (cell, option) with both a naming rate and a rating mean:
  overall     options named in >= 5% of the cell's responses; of those, how many are rated
              above / below zero; the same split per list; the AUC of rating mean as a
              separator of named (>= 5%) from unnamed options across all cells
  per cell    Spearman rank correlation between naming rate and rating mean across the
              cell's options, and the "exceptions": options named in >= 5% of
              responses whose mean rating is below zero
  exceptions  the full list of exceptions with their numbers (the "named but rated
              against" set), each carrying its named rate on the open ask and on the
              list's own ask
  by_ask      the overall and per-list tallies recomputed with the named rate taken from
              the open ask only ("open") and from the list's own ask only ("type"), and
              the exceptions that remain when only the open ask counts

    python3 analysis/concordance.py
"""
import json
import math
from collections import defaultdict

from common import DATA, LIST_ASK, LISTS

THRESH = 0.05
ASK_VIEWS = ("pooled", "open", "type")   # named rate over both asks, the open ask only, the list's own ask only


def named_rate(v, lst, view):
    if view == "pooled":
        return v["named_rate"]
    return v["named_by_ask"]["open" if view == "open" else LIST_ASK[lst]]


def spearman(x, y):
    n = len(x)

    def rank(v):
        s = sorted(range(n), key=lambda i: v[i]); r = [0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = rank(x), rank(y); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def main():
    cells = json.load((DATA / "cells.json").open())
    per_cell, exceptions = {}, []
    tot = defaultdict(lambda: {"named": 0, "rated_pos": 0, "rated_neg": 0, "rated_zero": 0})
    pos_scores, neg_scores = [], []   # for the AUC: rating means of named vs unnamed options
    for key, cell in cells.items():
        if key.startswith("_"):
            continue
        cond, lst = key.split("|")
        opts = [(o, v) for o, v in cell["options"].items() if not v["is_null"] and v["rating_mean"] is not None]
        x = [v["named_rate"] for _, v in opts]; y = [v["rating_mean"] for _, v in opts]
        exc = []
        for o, v in opts:
            named = v["named_rate"] >= THRESH
            (pos_scores if named else neg_scores).append(v["rating_mean"])
            if named:
                t = tot[lst]; t["named"] += 1
                if v["rating_mean"] > 0: t["rated_pos"] += 1
                elif v["rating_mean"] < 0:
                    t["rated_neg"] += 1
                    exc.append(o)
                    exceptions.append({"cell": key, "option": v["display"], "named_rate": round(v["named_rate"], 3),
                                       "named_rate_open": round(named_rate(v, lst, "open"), 3), "named_rate_type": round(named_rate(v, lst, "type"), 3),
                                       "advised_rate": round(v["advised_rate"], 3), "rating_mean": round(v["rating_mean"], 2),
                                       "n_models_named": v["n_models_named"], "n_models_advised": v["n_models_advised"]})
                else: t["rated_zero"] += 1
        per_cell[key] = {"n_options": len(opts), "rho": spearman(x, y) if len(opts) >= 5 else None,
                         "n_named": sum(1 for r in x if r >= THRESH), "exceptions": exc}
    auc_num = sum((p > q) + 0.5 * (p == q) for p in pos_scores for q in neg_scores)
    auc = auc_num / (len(pos_scores) * len(neg_scores))
    overall = {k: sum(t[k] for t in tot.values()) for k in ("named", "rated_pos", "rated_neg", "rated_zero")}
    rhos = sorted(v["rho"] for v in per_cell.values() if v["rho"] is not None)
    # the same tallies with the named rate taken from one ask only
    by_ask = {}
    for view in ("open", "type"):
        t_v = defaultdict(lambda: {"named": 0, "rated_pos": 0, "rated_neg": 0, "rated_zero": 0}); exc_v = []
        for key, cell in cells.items():
            if key.startswith("_"):
                continue
            lst = key.split("|")[1]
            for o, v in cell["options"].items():
                if v["is_null"] or v["rating_mean"] is None or named_rate(v, lst, view) < THRESH:
                    continue
                t = t_v[lst]; t["named"] += 1
                if v["rating_mean"] > 0: t["rated_pos"] += 1
                elif v["rating_mean"] < 0:
                    t["rated_neg"] += 1
                    exc_v.append({"cell": key, "option": v["display"], "named_rate_open": round(named_rate(v, lst, "open"), 3),
                                  "named_rate_type": round(named_rate(v, lst, "type"), 3), "rating_mean": round(v["rating_mean"], 2)})
                else: t["rated_zero"] += 1
        by_ask[view] = {"overall": {k: sum(t[k] for t in t_v.values()) for k in ("named", "rated_pos", "rated_neg", "rated_zero")},
                        "by_list": {l: t_v[l] for l in LISTS},
                        "exceptions": sorted(exc_v, key=lambda e: -e["named_rate_" + view])}
    out = {"threshold": THRESH, "overall": {**overall, "auc_rating_separates_named": auc,
                                             "n_pairs": len(pos_scores) + len(neg_scores),
                                             "rho_median": rhos[len(rhos) // 2], "rho_min": rhos[0], "n_cells_rho_positive": sum(r > 0 for r in rhos)},
           "by_list": {l: tot[l] for l in LISTS}, "by_ask": by_ask, "per_cell": per_cell,
           "exceptions": sorted(exceptions, key=lambda e: -e["named_rate"])}
    (DATA / "concordance.json").write_text(json.dumps(out, indent=1))
    o = out["overall"]
    print(f"options named >= 5%: {o['named']} | rated > 0: {o['rated_pos']} | rated < 0: {o['rated_neg']} | AUC {o['auc_rating_separates_named']:.3f} | rho median {o['rho_median']:.2f}, positive in {o['n_cells_rho_positive']}/{len(rhos)}")
    for l in LISTS:
        t = tot[l]; print(f"  {l:16s} named {t['named']:3d}  rated>0 {t['rated_pos']:3d}  rated<0 {t['rated_neg']:3d}")
    for view in ("open", "type"):
        b = by_ask[view]["overall"]; print(f"  {view + ' ask only':16s} named {b['named']:3d}  rated>0 {b['rated_pos']:3d}  rated<0 {b['rated_neg']:3d}  " + " ".join(f"{l[:5]} {by_ask[view]['by_list'][l]['rated_neg']}" for l in LISTS))
    print("top exceptions:", [(e["cell"], e["option"], e["named_rate"], e["rating_mean"]) for e in out["exceptions"][:6]])


if __name__ == "__main__":
    main()
