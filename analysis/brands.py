"""Frozen corpus -> data/brands.json: brand versus generic naming for three medications.

A deterministic string search over all 35,055 responses, independent of the judge's
extractions. Three medications from different drug classes, each with its best-known
brand: sertraline / Zoloft (SSRI), lisdexamfetamine / Vyvanse (stimulant), aripiprazole /
Abilify (antipsychotic). For each, per footing and overall: responses mentioning the generic
name, the brand name, both, and only one of them, plus the share of responses naming the
drug at all that carry the brand name. Whole-word, case-insensitive matches on the response
text; a match says the name appears, not that the drug was recommended.

    python3 analysis/brands.py [--runs <folder with runs/>]
"""
import argparse
import json
import re
from collections import defaultdict

from common import DATA, FOOTINGS, runs_root, iter_corpus

BRANDS = {"sertraline": "zoloft", "lisdexamfetamine": "vyvanse", "aripiprazole": "abilify"}
CLASS = {"sertraline": "SSRI", "lisdexamfetamine": "stimulant", "aripiprazole": "antipsychotic"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", "--dev", dest="runs"); args = ap.parse_args()
    runs_dir = runs_root(args.runs)
    pats = {g: (re.compile(rf"\b{g}\b", re.I), re.compile(rf"\b{b}\b", re.I)) for g, b in BRANDS.items()}
    acc = {g: {f: defaultdict(int) for f in FOOTINGS + ("all",)} for g in BRANDS}
    for r in iter_corpus(runs_dir):
        for g, (gre, bre) in pats.items():
            has_g, has_b = bool(gre.search(r["text"])), bool(bre.search(r["text"]))
            for f in (r["footing"], "all"):
                a = acc[g][f]; a["responses"] += 1
                if has_g or has_b:
                    a["either"] += 1; a["generic"] += has_g; a["brand"] += has_b
                    a["both"] += has_g and has_b; a["generic_only"] += has_g and not has_b; a["brand_only"] += has_b and not has_g
    out = {"note": "whole-word case-insensitive search of the response text; a match is a mention, not a recommendation",
           "medications": {}}
    for g in BRANDS:
        rows = {}
        for f in FOOTINGS + ("all",):
            a = acc[g][f]
            rows[f] = {**a, "brand_share_of_mentions": (a["brand"] / a["either"]) if a["either"] else None}
        out["medications"][g] = {"brand": BRANDS[g], "class": CLASS[g], "by_footing": rows}
    (DATA / "brands.json").write_text(json.dumps(out, indent=1))
    print(f"{'medication':16s} {'footing':10s} {'mention':>8s} {'generic':>8s} {'brand':>6s} {'both':>5s} {'brand only':>10s} {'brand share':>11s}")
    for g, v in out["medications"].items():
        for f in FOOTINGS + ("all",):
            a = v["by_footing"][f]
            print(f"{g + '/' + v['brand']:16s} {f:10s} {a['either']:8d} {a['generic']:8d} {a['brand']:6d} {a['both']:5d} {a['brand_only']:10d} {100 * a['brand_share_of_mentions']:10.0f}%")


if __name__ == "__main__":
    main()
