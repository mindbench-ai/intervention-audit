"""responses.jsonl + entities.jsonl -> data/mentions.json: how many interventions a response
names, and with what stance, by any breakdown.

Keys: "all", "footing|<f>", "ask|<a>", "footing_ask|<f>|<a>", "condition|<c>",
"ask_condition|<a>|<c>", "family|<fam>", "model|<m>", "model_footing|<m>|<f>", "model_ask|<m>|<a>". Each holds
n (responses); per_response: mean and median of extracted mentions, recommended mentions,
advised-against mentions, and recommended mentions that map to a rating-list option;
stance: the slice's mention count and its four-way and binary stance shares (common.py
definition); mapped_share: share of the slice's mentions that map to a rating-list option.

    python3 analysis/mentions.py
"""
import json
import statistics
from collections import Counter, defaultdict

from common import DATA, read_jsonl

FIELDS = ("n_entities", "n_recommended", "n_advised_against", "n_recommended_listed")
STANCES = ("recommended", "neutral", "mixed", "advised_against")


def keys(r):
    f, a, c, m = r["footing"], r["ask"], r["condition"], r["model"]
    return ("all", f"footing|{f}", f"ask|{a}", f"footing_ask|{f}|{a}", f"condition|{c}",
            f"ask_condition|{a}|{c}", f"family|{r['family']}", f"model|{m}", f"model_footing|{m}|{f}", f"model_ask|{m}|{a}")


def main():
    per = defaultdict(lambda: {f: [] for f in FIELDS})
    for r in read_jsonl(DATA / "responses.jsonl"):
        for k in keys(r):
            for f in FIELDS:
                per[k][f].append(r[f])
    st = defaultdict(Counter)
    for e in read_jsonl(DATA / "entities.jsonl"):
        for k in keys(e):
            c = st[k]; c["n"] += 1; c[e["stance"]] += 1; c["mapped"] += e["option"] is not None
    out = {}
    for k, v in per.items():
        n = len(v["n_entities"]); c = st[k]
        out[k] = {"n": n,
                  "per_response": {f: {"mean": statistics.fmean(v[f]), "median": statistics.median(v[f])} for f in FIELDS},
                  "stance": {"n_mentions": c["n"],
                             "four_way": {s: c[s] / c["n"] for s in STANCES},
                             "binary": {"recommended": (c["n"] - c["advised_against"]) / c["n"],
                                        "advised_against": c["advised_against"] / c["n"]}},
                  "mapped_share": c["mapped"] / c["n"]}
    (DATA / "mentions.json").write_text(json.dumps(out, indent=1))
    show(out, ["all"] + [f"footing|{f}" for f in ("clinical", "caregiver", "patient")])


def show(out, ks):
    print(f"{'slice':18s} {'resp':>6s} | {'extracted':>13s} {'recommended':>13s} {'advised':>13s} {'rec. listed':>13s} | "
          f"{'rec':>5s} {'neut':>5s} {'mix':>5s} {'adv':>5s} | {'mapped':>6s}")
    print(" " * 27 + "mean / median" + " " * 40 + "four-way stance shares")
    for k in ks:
        x = out[k]; p = x["per_response"]; s = x["stance"]["four_way"]
        cells = "  ".join(f"{p[f]['mean']:5.1f} / {p[f]['median']:4.0f}" for f in FIELDS)
        print(f"{k:18s} {x['n']:6d} | {cells} | " + " ".join(f"{s[t]:5.1%}" for t in STANCES) + f" | {x['mapped_share']:6.1%}")


if __name__ == "__main__":
    main()
