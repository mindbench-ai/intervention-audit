"""Base tables -> data/cells.json: the per-cell, per-option numbers behind every figure.

A cell is one condition and one intervention list. For each option on the list:
  named_rate        share of the cell's responses (matching ask + open ask, all footings,
                    all samples, all 41 models: 738 responses) in which the option is named
                    and counted as recommended (binary stance; see common.py)
  advised_rate      share in which it is advised against
  named_by_footing, advised_by_footing
  named_by_ask, advised_by_ask   the same split by the two asks that feed the cell: "open" and the
                    list's own ask (medication / psychotherapy / neuromodulation / digital)
  n_models_named, n_models_advised   models (of 41) doing so at least once in the cell
  rating_hist       counts of scores -3..3 across the cell's rating calls (29 models)
  rating_mean, rating_n, rating_by_footing
  is_null           the list's null option (rating arm only)
Plus "_summary": for all 19 presentations and for the 17 condition vignettes (no controls), per list: the
number of options named by at least 5% of responses (median, IQR, range), the top option's
named rate (median, range), the number of cells whose top option reaches 50%, and which
options are top; "_all" holds the mapped share and the denominators used.

    python3 analysis/cells.py
"""
import json
from collections import Counter, defaultdict
from statistics import median, quantiles

from common import (CONDITIONS, CONTROLS, DATA, FOOTINGS, LISTS, LIST_ASK, MODELS, NULLS, OPTIONS, OPTION_DISPLAY,
                    read_jsonl)


def main():
    # generation side: per cell, per option, the set of responses naming it (rec / adv), by footing
    named = defaultdict(lambda: defaultdict(set))    # (cond, list, option) -> footing -> {rid}
    advised = defaultdict(lambda: defaultdict(set))
    named_models = defaultdict(set); advised_models = defaultdict(set)
    n_resp = defaultdict(int); n_resp_footing = defaultdict(int); n_resp_ask = defaultdict(int)
    named_ask = defaultdict(lambda: defaultdict(set)); advised_ask = defaultdict(lambda: defaultdict(set))  # (cond, list, option) -> ask -> {rid}
    n_mentions = n_mapped = 0
    for r in read_jsonl(DATA / "responses.jsonl"):
        for lst in LISTS:
            if r["ask"] in (LIST_ASK[lst], "open"):
                n_resp[(r["condition"], lst)] += 1
                n_resp_footing[(r["condition"], lst, r["footing"])] += 1
                n_resp_ask[(r["condition"], lst, r["ask"])] += 1
    for e in read_jsonl(DATA / "entities.jsonl"):
        n_mentions += 1
        if not e["option"]:
            continue
        n_mapped += 1
        lst = e["list"]
        if e["ask"] not in (LIST_ASK[lst], "open"):
            continue
        key = (e["condition"], lst, e["option"])
        if e["binary"] == "recommended":
            named[key][e["footing"]].add(e["rid"]); named_models[key].add(e["model"]); named_ask[key][e["ask"]].add(e["rid"])
        else:
            advised[key][e["footing"]].add(e["rid"]); advised_models[key].add(e["model"]); advised_ask[key][e["ask"]].add(e["rid"])
    # rating side
    hist = defaultdict(lambda: [0] * 7); rsum = defaultdict(float); rn = defaultdict(int)
    rsum_f = defaultdict(float); rn_f = defaultdict(int)
    for r in read_jsonl(DATA / "ratings.jsonl"):
        key = (r["condition"], r["list"], r["option"])
        hist[key][r["score"] + 3] += 1; rsum[key] += r["score"]; rn[key] += 1
        rsum_f[(key, r["footing"])] += r["score"]; rn_f[(key, r["footing"])] += 1

    cells = {}
    for cond in CONDITIONS:
        for lst in LISTS:
            opts = {}
            denom = n_resp[(cond, lst)]
            for opt in OPTIONS[lst] + [NULLS[lst]]:
                key = (cond, lst, opt)
                nm = set().union(*named[key].values()) if named[key] else set()
                ad = set().union(*advised[key].values()) if advised[key] else set()
                row = {"display": OPTION_DISPLAY.get((lst, opt), opt), "is_null": opt == NULLS[lst],
                       "named_rate": len(nm) / denom, "advised_rate": len(ad) / denom,
                       "named_by_footing": {f: len(named[key][f]) / n_resp_footing[(cond, lst, f)] for f in FOOTINGS},
                       "advised_by_footing": {f: len(advised[key][f]) / n_resp_footing[(cond, lst, f)] for f in FOOTINGS},
                       "named_by_ask": {a: len(named_ask[key][a]) / n_resp_ask[(cond, lst, a)] for a in ("open", LIST_ASK[lst])},
                       "advised_by_ask": {a: len(advised_ask[key][a]) / n_resp_ask[(cond, lst, a)] for a in ("open", LIST_ASK[lst])},
                       "n_models_named": len(named_models[key]), "n_models_advised": len(advised_models[key]),
                       "rating_hist": hist[key] if rn[key] else None,
                       "rating_mean": rsum[key] / rn[key] if rn[key] else None, "rating_n": rn[key],
                       "rating_by_footing": {f: (rsum_f[(key, f)] / rn_f[(key, f)] if rn_f[(key, f)] else None) for f in FOOTINGS}}
                opts[opt] = row
            cells[f"{cond}|{lst}"] = {"n_responses": denom, "options": opts}

    summary = {}
    for scope, conds in (("all", CONDITIONS), ("clinical", [c for c in CONDITIONS if c not in CONTROLS])):
        summary[scope] = {}
        for lst in LISTS:
            above5, top, top_opt, half = [], [], Counter(), []
            for cond in conds:
                opts = {k: o["named_rate"] for k, o in cells[f"{cond}|{lst}"]["options"].items() if not o["is_null"]}
                above5.append(sum(1 for x in opts.values() if x >= 0.05)); top.append(max(opts.values()))
                top_opt[max(opts, key=opts.get)] += 1
                if max(opts.values()) >= 0.5:
                    half.append(f"{cond}|{max(opts, key=opts.get)}")
            q = quantiles(above5, n=4)
            summary[scope][lst] = {"n_cells": len(conds), "n_options": len(OPTIONS[lst]),
                                   "options_above_5pct": {"median": median(above5), "iqr": [q[0], q[2]], "min": min(above5), "max": max(above5)},
                                   "top_option_rate": {"median": median(top), "min": min(top), "max": max(top)},
                                   "n_cells_top_at_least_50pct": len(half), "cells_top_at_least_50pct": half,
                                   "top_option_counts": dict(top_opt.most_common())}
    summary["_all"] = {"n_mentions": n_mentions, "n_mapped": n_mapped, "mapped_share": n_mapped / n_mentions,
                       "n_models_generation": len(MODELS), "responses_per_cell": n_resp[(CONDITIONS[0], LISTS[0])]}
    cells["_summary"] = summary
    (DATA / "cells.json").write_text(json.dumps(cells, indent=1))
    print(f"cells: {len(cells) - 1} | responses per cell: {summary['_all']['responses_per_cell']} | mapped share {summary['_all']['mapped_share']:.1%}")
    for scope in ("all", "clinical"):
        for lst in LISTS:
            s = summary[scope][lst]
            print(f"  {scope:8s} {lst:16s} options >=5%: median {s['options_above_5pct']['median']:.0f} (IQR {s['options_above_5pct']['iqr'][0]:.0f}-{s['options_above_5pct']['iqr'][1]:.0f}) | "
                  f"top rate median {s['top_option_rate']['median']:.0%} ({s['top_option_rate']['min']:.0%}-{s['top_option_rate']['max']:.0%}) | cells with top >=50%: {s['n_cells_top_at_least_50pct']}/{s['n_cells']}")


if __name__ == "__main__":
    main()
