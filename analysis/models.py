"""mentions.json + entities.jsonl + ratings.jsonl + cells.json -> data/models.json: per-model
behavior on both arms.

Per model (41 generation models; 29 also rated):
  breadth        recommended interventions per response, all asks and open ask only (from
                 mentions.json); advised-against per response; distinct list options named
                 per cell (mean over the 76 cells)
  spray          share of the model's list-mapped recommendations (over a cell's two asks) whose
                 panel mean rating is below zero, and the same on the open ask only. The panel
                 mean is leave-one-out for rated models: the model's own ratings are excluded
  consensus      share of the model's list-mapped recommendations that are the cell's most-named
                 option across all 41 models
  self           rated models only: over the model's own list-mapped recommendations that it also
                 rated, the share it itself rates below zero and the mean of its own ratings of
                 them, over the cell's two asks and on the open ask only
  judgement      rated models only: Spearman rho between the model's own option ratings and the
                 leave-one-out panel mean over all (cell, option) pairs; median per-cell Spearman
                 rho between the model's own naming counts and its own ratings
  controls       rated models only: mean rating of the null option on the two watchful-waiting
                 controls and the share of those calls in which the null option ranked first
Per family: means over its models. "_summary": Spearman rho across models between breadth
and the spray share, with the range of each.

    python3 analysis/models.py
"""
import json
import math
import statistics
from collections import defaultdict

from common import CONTROLS, DATA, FAMILY, LIST_ASK, LISTS, MODELS, read_jsonl


def spearman(x, y):
    n = len(x)
    if n < 3:
        return None

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
    mentions = json.load((DATA / "mentions.json").open())
    cells = json.load((DATA / "cells.json").open())
    top = {}
    for key, cell in cells.items():
        if key.startswith("_"):
            continue
        cond, lst = key.split("|")
        top[(cond, lst)] = max((o for o, v in cell["options"].items() if not v["is_null"]),
                               key=lambda o: cell["options"][o]["named_rate"])

    # ratings: panel sums per (cond, list, option) and per model
    S = defaultdict(float); N = defaultdict(int)
    own_s = defaultdict(lambda: defaultdict(float)); own_n = defaultdict(lambda: defaultdict(int))
    calls = defaultdict(dict)
    for r in read_jsonl(DATA / "ratings.jsonl"):
        m = r["model"]
        if r["condition"] in CONTROLS:
            calls[(m, r["condition"], r["footing"], r["list"], r["sample"])][r["option"]] = r["score"]
        if r["is_null"]:
            continue
        k = (r["condition"], r["list"], r["option"])
        S[k] += r["score"]; N[k] += 1; own_s[m][k] += r["score"]; own_n[m][k] += 1
    rated = set(own_s)

    def loo_mean(k, m):
        s, n = S[k] - own_s[m].get(k, 0.0), N[k] - own_n[m].get(k, 0)
        return s / n if n else None

    # generation: list-mapped recommendations over the cell's two asks
    mapped = defaultdict(int); mapped_open = defaultdict(int)
    neg = defaultdict(int); neg_open = defaultdict(int); hit_top = defaultdict(int)
    named_opts = defaultdict(lambda: defaultdict(set))          # model -> cell -> options named
    named_n = defaultdict(lambda: defaultdict(int))             # model -> (cond, list, option) -> mentions
    self_acc = defaultdict(lambda: {"n": 0, "neg": 0, "sum": 0.0, "n_open": 0, "neg_open": 0, "sum_open": 0.0})
    for e in read_jsonl(DATA / "entities.jsonl"):
        if not e["option"] or e["binary"] != "recommended":
            continue
        lst = e["list"]
        if e["ask"] not in ("open", LIST_ASK[lst]):
            continue
        m = e["model"]; k = (e["condition"], lst, e["option"])
        pm = loo_mean(k, m) if N[k] else None
        is_neg = pm is not None and pm < 0
        mapped[m] += 1; neg[m] += is_neg
        if e["ask"] == "open":
            mapped_open[m] += 1; neg_open[m] += is_neg
        hit_top[m] += e["option"] == top[(e["condition"], lst)]
        named_opts[m][(e["condition"], lst)].add(e["option"]); named_n[m][k] += 1
        if own_n[m].get(k):
            so = own_s[m][k] / own_n[m][k]; a = self_acc[m]
            a["n"] += 1; a["neg"] += so < 0; a["sum"] += so
            if e["ask"] == "open":
                a["n_open"] += 1; a["neg_open"] += so < 0; a["sum_open"] += so

    # controls: null option
    null_acc = defaultdict(lambda: {"null": [], "first": 0, "n": 0})
    for (m, *_), scores in calls.items():
        nulls = [v for k, v in scores.items() if k.startswith("no ")]
        if nulls:
            a = null_acc[m]; a["null"].append(nulls[0]); a["n"] += 1; a["first"] += nulls[0] >= max(scores.values())

    mean = lambda v: round(sum(v) / len(v), 12) if v else None  # noqa: E731  rounded so the output is bit-stable
    out = {"models": {}, "families": {}}
    for m in MODELS:
        pr = mentions[f"model|{m}"]["per_response"]
        row = {"family": FAMILY[m], "rated": m in rated,
               "rec_per_response": pr["n_recommended"]["mean"],
               "rec_per_response_open": mentions[f"model_ask|{m}|open"]["per_response"]["n_recommended"]["mean"],
               "adv_per_response": pr["n_advised_against"]["mean"],
               "options_named_per_cell": mean([len(named_opts[m][c]) for c in top]),
               "n_listed_recommendations": mapped[m],
               "share_rated_below_zero": neg[m] / mapped[m],
               "share_rated_below_zero_open": neg_open[m] / mapped_open[m],
               "share_top_option": hit_top[m] / mapped[m],
               "rating_vs_panel_rho": None, "naming_vs_own_rating_rho_median": None,
               "self_share_rated_below_zero": None, "self_share_rated_below_zero_open": None,
               "self_mean_rating_of_recs": None, "self_mean_rating_of_recs_open": None, "n_self_rated_recs": 0,
               "null_rating_controls": None, "null_ranked_first_controls": None, "n_control_calls": 0}
        if m in rated:
            keys = [k for k in own_s[m] if N[k] - own_n[m][k] > 0]
            row["rating_vs_panel_rho"] = spearman([own_s[m][k] / own_n[m][k] for k in keys], [loo_mean(k, m) for k in keys])
            rhos = []
            for (cond, lst) in top:
                ks = [k for k in own_s[m] if k[0] == cond and k[1] == lst]
                x = [named_n[m][k] for k in ks]
                if len(ks) >= 5 and len(set(x)) >= 2:
                    rho = spearman(x, [own_s[m][k] / own_n[m][k] for k in ks])
                    if rho is not None:
                        rhos.append(rho)
            row["naming_vs_own_rating_rho_median"] = statistics.median(rhos) if rhos else None
            a = self_acc[m]
            row.update(self_share_rated_below_zero=a["neg"] / a["n"], self_share_rated_below_zero_open=a["neg_open"] / a["n_open"],
                       self_mean_rating_of_recs=a["sum"] / a["n"], self_mean_rating_of_recs_open=a["sum_open"] / a["n_open"],
                       n_self_rated_recs=a["n"])
            a = null_acc[m]
            row.update(null_rating_controls=mean(a["null"]), null_ranked_first_controls=a["first"] / a["n"] if a["n"] else None,
                       n_control_calls=a["n"])
        out["models"][m] = row
    fams = defaultdict(list)
    for v in out["models"].values():
        fams[v["family"]].append(v)
    for f, vs in fams.items():
        out["families"][f] = {"n_models": len(vs), "n_rated": sum(v["rated"] for v in vs),
                              **{k: mean([v[k] for v in vs if v[k] is not None]) for k in
                                 ("rec_per_response", "rec_per_response_open", "adv_per_response", "share_rated_below_zero",
                                  "share_rated_below_zero_open", "share_top_option", "rating_vs_panel_rho",
                                  "self_share_rated_below_zero", "self_share_rated_below_zero_open", "null_rating_controls")}}
    ms = out["models"]
    x = [v["rec_per_response"] for v in ms.values()]; y = [v["share_rated_below_zero"] for v in ms.values()]
    yo = [v["share_rated_below_zero_open"] for v in ms.values()]
    rr = [v["rating_vs_panel_rho"] for v in ms.values() if v["rating_vs_panel_rho"] is not None]
    sv = [v["self_share_rated_below_zero"] for v in ms.values() if v["rated"]]
    svo = [v["self_share_rated_below_zero_open"] for v in ms.values() if v["rated"]]
    xr = [v["rec_per_response"] for v in ms.values() if v["rated"]]
    out["_summary"] = {"n_models": len(ms), "n_rated": len(rr),
                       "self_share_below_zero_range": [min(sv), max(sv)], "self_share_below_zero_open_range": [min(svo), max(svo)],
                       "rho_breadth_vs_self_share_below_zero": spearman(xr, sv),
                       "rho_self_vs_panel_share_below_zero": spearman(sv, [v["share_rated_below_zero"] for v in ms.values() if v["rated"]]),
                       "rho_breadth_vs_share_below_zero": spearman(x, y), "rho_breadth_vs_share_below_zero_open": spearman(x, yo),
                       "rec_per_response_range": [min(x), max(x)], "share_below_zero_range": [min(y), max(y)],
                       "share_below_zero_open_range": [min(yo), max(yo)],
                       "rating_vs_panel_rho_range": [min(rr), max(rr)], "rating_vs_panel_rho_median": statistics.median(rr)}
    (DATA / "models.json").write_text(json.dumps(out, indent=1))
    s = out["_summary"]
    print(f"{len(ms)} models: recommended/response {s['rec_per_response_range'][0]:.1f}-{s['rec_per_response_range'][1]:.1f}; "
          f"share rated below 0 (LOO) {100*s['share_below_zero_range'][0]:.0f}-{100*s['share_below_zero_range'][1]:.0f}% pooled, "
          f"{100*s['share_below_zero_open_range'][0]:.0f}-{100*s['share_below_zero_open_range'][1]:.0f}% open ask; "
          f"rho breadth~share {s['rho_breadth_vs_share_below_zero']:.2f} pooled, {s['rho_breadth_vs_share_below_zero_open']:.2f} open")
    print(f"rated {s['n_rated']}: own rating vs LOO panel rho {s['rating_vs_panel_rho_range'][0]:.2f}-{s['rating_vs_panel_rho_range'][1]:.2f} (median {s['rating_vs_panel_rho_median']:.2f})")
    print(f"self-rated below 0: {100*s['self_share_below_zero_range'][0]:.0f}-{100*s['self_share_below_zero_range'][1]:.0f}% pooled, "
          f"{100*s['self_share_below_zero_open_range'][0]:.0f}-{100*s['self_share_below_zero_open_range'][1]:.0f}% open; "
          f"rho breadth~self {s['rho_breadth_vs_self_share_below_zero']:.2f}; rho self~panel {s['rho_self_vs_panel_share_below_zero']:.2f}")
    print(f"{'model':28s} {'rec':>5s} {'open':>5s} {'adv':>4s} {'opt/cell':>8s} {'<0%':>5s} {'<0% open':>8s} {'top%':>5s} {'rho panel':>9s} {'self<0%':>7s} {'self<0% open':>12s}")
    for m, v in sorted(ms.items(), key=lambda kv: -kv[1]["rec_per_response"]):
        rp = f"{v['rating_vs_panel_rho']:.2f}" if v["rating_vs_panel_rho"] is not None else ""
        sf = f"{100*v['self_share_rated_below_zero']:6.1f}%" if v["rated"] else ""
        sfo = f"{100*v['self_share_rated_below_zero_open']:11.1f}%" if v["rated"] else ""
        print(f"{m:28s} {v['rec_per_response']:5.1f} {v['rec_per_response_open']:5.1f} {v['adv_per_response']:4.1f} {v['options_named_per_cell']:8.1f} "
              f"{100*v['share_rated_below_zero']:4.1f}% {100*v['share_rated_below_zero_open']:7.1f}% {100*v['share_top_option']:4.0f}% {rp:>9s} {sf:>7s} {sfo:>12s}")


if __name__ == "__main__":
    main()
