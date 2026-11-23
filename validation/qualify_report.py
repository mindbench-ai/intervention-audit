"""Score the rating-arm qualification probe against the pre-specified gate.

A model qualifies for the structured-rating arm if at least 80% of its probe calls cover
at least 90% of the options (the parser accepts only in-range integer scores, so scale
validity folds into coverage). The gate reads format compliance only, never rating
content. Errored calls never tested the model and are excluded, but a model needs at least
20 of 24 valid cells to be judged at all; the latest row per cell wins (re-probes
supersede starved or errored attempts).

Also prints the measured cost projection for the full rating grid (684 calls per model)
from mean probe tokens and the list prices in config/models.json.

Reads runs/qualify/probe.jsonl from the raw runs. Writes
validation/qualification_verdicts.json.

    python3 validation/qualify_report.py [--run runs/qualify]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL_GRID = 19 * 3 * 3 * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/qualify")
    args = ap.parse_args()
    cells = {}
    for line in (ROOT / args.run / "probe.jsonl").open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cells[(r["model_key"], r["condition"], r["scope"], r["sample"])] = r
    rows = defaultdict(list)
    for r in cells.values():
        rows[r["model_key"]].append(r)
    prices = {k: v.get("price_usd_per_1m") for k, v in json.loads((ROOT / "config/models.json").read_text())["models"].items()}

    verdicts = {}
    print(f"{'model':26} {'pass-rate':>9} {'med-cov':>8} {'verdict':>20} {'est. rating cost':>16}")
    for mk, rs in sorted(rows.items()):
        ok = [r for r in rs if not r.get("error")]
        passing = sum(1 for r in ok if r["coverage"] >= 0.90)
        rate = passing / len(ok) if len(ok) >= 20 else -1
        cov = sorted(r["coverage"] for r in ok)
        med = cov[len(cov) // 2] if cov else 0
        verdict = ("INCOMPLETE - re-probe" if rate < 0 else "qualified" if rate >= 0.80 else "demoted (gen-only)")
        tin = [r["tokens_in"] for r in ok if r.get("tokens_in")]
        tout = [r["tokens_out"] for r in ok if r.get("tokens_out")]
        cost = None
        if tin and tout and prices.get(mk):
            p = prices[mk]
            cost = FULL_GRID * (sum(tin) / len(tin) * p["input"] + sum(tout) / len(tout) * p["output"]) / 1e6
        verdicts[mk] = {"verdict": verdict, "pass_rate": max(rate, 0), "median_coverage": med,
                        "n_valid_cells": len(ok), "est_rating_cost_usd": round(cost, 2) if cost is not None else None}
        print(f"{mk:26} {max(rate, 0):9.0%} {med:8.0%} {verdict:>20} {('$%7.2f' % cost) if cost is not None else '':>16}")
    (ROOT / "validation/qualification_verdicts.json").write_text(json.dumps(verdicts, indent=1) + "\n")
    print(f"\n{sum(1 for v in verdicts.values() if v['verdict'] == 'qualified')}/{len(verdicts)} qualified; wrote validation/qualification_verdicts.json")


if __name__ == "__main__":
    main()
