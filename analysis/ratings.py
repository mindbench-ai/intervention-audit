"""Rating runs -> data/ratings.jsonl: one row per (rating call, option).

Rows: model, family, language, pack_version, condition, footing, list, sample, option (lower-case option name or
the null option), score (-3..3), is_null, source (main | probe).

Cell rule: the main rating run (runs/rating_v1) supplies every cell it holds, keeping the
latest usable row per cell (re-runs on resume supersede earlier attempts). The
qualification probe (runs/qualify) supplies only the cells the main run does not hold;
the probe was the same instrument on 24 cells per model and those cells were counted as
done rather than re-run. Rows with errors or no parsed ratings are skipped. Grid cells
(29 models x 19 x 3 x 4 x 3) that end with no usable row are written to
data/rating_gaps.json with every attempt's finish reason and error, so the gap is on record.

    python3 analysis/ratings.py [--runs <folder with runs/>]
"""
import argparse
import glob
import json
from collections import Counter

from common import CONDITIONS, DATA, FAMILY, FOOTINGS, LANGUAGE, LISTS, NULLS, runs_root, write_jsonl


def usable(r):
    return not r.get("error") and bool(r.get("ratings")) and r.get("response")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", "--dev", dest="runs")
    args = ap.parse_args()
    runs_dir = runs_root(args.runs)
    cells = {}   # (model, condition, footing, list, sample) -> (source, ratings, pack_version)
    attempts = {}  # every main-run attempt per cell: (finish_reason, error_type, error)
    for f in sorted(glob.glob(str(runs_dir / "runs/rating_v1/*/ratings.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            key = (r["model_key"], r["condition"], r["register"], r["scope"], r["sample"])
            attempts.setdefault(key, []).append({"finish_reason": r.get("finish_reason"), "error_type": r.get("error_type"),
                                                 "error": (r.get("error") or None) and str(r.get("error"))[:120], "n_rated": r.get("n_rated")})
            if usable(r):
                cells[key] = ("main", r["ratings"], r.get("pack_version"))
    n_main = len(cells)
    for line in (runs_dir / "runs/qualify/probe.jsonl").open():
        r = json.loads(line)
        if not usable(r):
            continue
        key = (r["model_key"], r["condition"], r.get("register", "clinical"), r["scope"], r["sample"])
        if key not in cells:
            cells[key] = ("probe", r["ratings"], r.get("pack_version"))
    rows = []
    for (model, cond, footing, lst, sample), (source, ratings, pack) in sorted(cells.items()):
        null_name = NULLS[lst]
        for opt, score in ratings.items():
            rows.append({"model": model, "family": FAMILY.get(model), "language": LANGUAGE, "pack_version": pack,
                         "condition": cond, "footing": footing,
                         "list": lst, "sample": sample, "option": opt.lower(), "score": int(score),
                         "is_null": opt.lower() == null_name, "source": source})
    n = write_jsonl(DATA / "ratings.jsonl", rows)
    rated_models = sorted({k[0] for k in cells})
    samples = sorted({k[4] for k in cells})   # sample ids as stored in the run (0, 1, 2)
    gaps = []
    for m in rated_models:
        for cond in CONDITIONS:
            for f in FOOTINGS:
                for lst in LISTS:
                    for smp in samples:
                        key = (m, cond, f, lst, smp)
                        if key not in cells:
                            gaps.append({"model": m, "condition": cond, "footing": f, "list": lst, "sample": smp,
                                         "attempts": attempts.get(key, [])})
    (DATA / "rating_gaps.json").write_text(json.dumps({"grid_cells": len(rated_models) * len(CONDITIONS) * 3 * 4 * 3,
                                                        "cells_with_ratings": len(cells), "gaps": gaps}, indent=1, default=str))
    src = Counter(v[0] for v in cells.values())
    per_model = Counter(k[0] for k in cells)
    print(f"cells: {len(cells)} (main {src['main']}, probe-only {src['probe']}; main run held {n_main}) | option rows: {n}")
    print(f"models: {len(per_model)} | cells per model: min {min(per_model.values())} max {max(per_model.values())}")
    for g in gaps:
        print(f"gap: {g['model']}|{g['condition']}|{g['footing']}|{g['list']}|s{g['sample']} attempts: "
              + "; ".join(f"{a['finish_reason'] or a['error_type']}" for a in g["attempts"]))


if __name__ == "__main__":
    main()
