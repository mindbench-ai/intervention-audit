"""Corpus-level counts -> data/corpus.json.

What is in the dataset, independent of any analysis: responses and rating calls by arm,
models per arm, the grid each model completed, extracted mentions and their four-way
stance distribution, the two-way distribution under the recommended / advised-against
definition (common.py), and the share of mentions that map to a rating-list option.

    python3 analysis/corpus.py
"""
import json
from collections import Counter

from common import DATA, LISTS, read_jsonl


def main():
    resp = list(read_jsonl(DATA / "responses.jsonl"))
    models_gen = sorted({r["model"] for r in resp})
    stances, binary = Counter(), Counter()
    n_ent = n_mapped = 0
    per_list = Counter()
    for e in read_jsonl(DATA / "entities.jsonl"):
        n_ent += 1
        stances[e["stance"]] += 1
        binary[e["binary"]] += 1
        if e["option"]:
            n_mapped += 1
            per_list[e["list"]] += 1
    cells = set(); models_rate = set(); sources = Counter()
    seen = set()
    for r in read_jsonl(DATA / "ratings.jsonl"):
        key = (r["model"], r["condition"], r["footing"], r["list"], r["sample"])
        if key not in seen:
            seen.add(key); sources[r["source"]] += 1
        cells.add(key); models_rate.add(r["model"])
    gaps = json.load((DATA / "rating_gaps.json").open())["gaps"]
    out = {
        "generation": {"responses": len(resp), "models": len(models_gen), "responses_per_model": len(resp) // len(models_gen),
                       "grid": "19 conditions x 3 footings x 5 asks x 3 samples"},
        "rating": {"calls_with_ratings": len(cells), "models": len(models_rate), "calls_per_model_grid": 19 * 3 * 4 * 3,
                   "grid_total": len(models_rate) * 19 * 3 * 4 * 3, "from_main_run": sources["main"], "from_probe_only": sources["probe"],
                   "grid": "19 conditions x 3 footings x 4 lists x 3 samples",
                   "cells_without_ratings": [{"cell": f"{g['model']}|{g['condition']}|{g['footing']}|{g['list']}|s{g['sample']}",
                                              "finish_reasons": sorted({a["finish_reason"] for a in g["attempts"] if a["finish_reason"]}),
                                              "n_attempts": len(g["attempts"])} for g in gaps]},
        "total_model_outputs": len(resp) + len(cells),
        "mentions": {"total": n_ent, "four_way": {k: {"n": v, "share": v / n_ent} for k, v in stances.most_common()},
                     "binary": {k: {"n": v, "share": v / n_ent} for k, v in binary.most_common()},
                     "mapped_to_list": {"n": n_mapped, "share": n_mapped / n_ent, "by_list": dict(per_list)}},
    }
    (DATA / "corpus.json").write_text(json.dumps(out, indent=1) + "\n")
    g, r, m = out["generation"], out["rating"], out["mentions"]
    print(f"generation: {g['responses']} responses from {g['models']} models | rating: {r['calls_with_ratings']} calls with ratings from {r['models']} models (grid {r['grid_total']}) | total {out['total_model_outputs']}")
    print("mentions:", m["total"], "| four-way:", {k: f"{v['share']:.1%}" for k, v in m["four_way"].items()}, "| binary:", {k: f"{v['share']:.1%}" for k, v in m["binary"].items()}, f"| mapped {m['mapped_to_list']['share']:.1%}")


if __name__ == "__main__":
    main()
