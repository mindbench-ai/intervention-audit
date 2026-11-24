#!/usr/bin/env python3
"""Download a release of the MindBench Intervention Audit dataset from Hugging Face and
restore the local files this repository reads.

    python3 data/fetch.py                      # data/entities.jsonl and data/ratings.jsonl
    python3 data/fetch.py --runs <folder>      # also rebuild the raw run store in <folder>
    python3 data/fetch.py --revision v1.0.0    # a tagged release (the default)
    python3 data/fetch.py --source <folder>    # a local build instead of Hugging Face

The two derived tables are written to data/ and checked against data/MANIFEST.json. The run
store holds the responses, the judge output, and every rating attempt in the layout the
analysis chain reads, so that

    python3 analysis/make_all.py --runs <folder> --check

rebuilds every table in data/ from the released dataset. Downloads are cached under
data/.cache/<revision>/. Standard library only.
"""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPO = "mindbench-ai/intervention-audit"
CONFIGS = ("responses", "judgments", "rating_calls", "entities", "ratings", "rating_attempts")


def config_path(name, args):
    rel = f"data/{name}/train.jsonl"
    if args.source:
        return Path(args.source) / rel
    cached = DATA / ".cache" / args.revision / rel
    if not cached.exists():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{args.revision}/{rel}"
        print(f"  downloading {name} ...")
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.rename(cached)
    return cached


def rows(name, args):
    with open(config_path(name, args), encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def restore_table(name, args, dest):
    """Write a derived table byte for byte and return its sha256."""
    h = hashlib.sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path(name, args), "rb") as src, open(dest, "wb") as out:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            out.write(chunk)
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in items:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def item_key(r):
    return f"{r['condition']}|{r['footing']}|{r['ask']}|{r['sample']}"


def rebuild_runs(args, store):
    """Lay the released rows out as the run files analysis/common.py and analysis/ratings.py
    read. Row order follows the dataset, which is the order the analysis read the original
    runs in."""
    # generation corpus: one file per model, numbered so that sorted order is dataset order
    side = store / "runs/gen_v1/export/sidecar"
    fh, current, index, n = None, None, 0, 0
    for r in rows("responses", args):
        if r["model"] != current:
            if fh:
                fh.close()
            current, index = r["model"], index + 1
            side.mkdir(parents=True, exist_ok=True)
            fh = open(side / f"items-{index:03d}-{current}.jsonl", "w")
        fh.write(json.dumps({"item_key": item_key(r), "raw": {"model_key": r["model"]},
                             "output": r["response"]}, ensure_ascii=False) + "\n")
        n += 1
    if fh:
        fh.close()
    print(f"  responses     {n:>9,d} rows -> runs/gen_v1/export/sidecar/")

    out = []
    for r in rows("judgments", args):
        out.append({"model_key": r["model"], "item_key": item_key(r), "condition": r["condition"],
                    "register": r["footing"], "category": r["ask"], "sample": r["sample"],
                    "annotation": {"entities": r["entities"], "consult_professional": r["consult_professional"],
                                   "not_a_doctor": r["not_a_doctor"]},
                    "error": None, "judge_model": r["judge_model"], "prompt_sha256": r["judge_prompt_sha256"],
                    "pack_version": r["pack_version"], "ts": r["ts"]})
    write_jsonl(store / "runs/judge_v1/luna/judgments.jsonl", out)
    print(f"  judgments     {len(out):>9,d} rows -> runs/judge_v1/luna/judgments.jsonl")

    main, probe = [], []
    for r in rows("rating_attempts", args):
        row = {"model_key": r["model"], "model_id": r["model_id"], "served_model": r["served_model"],
               "pack_version": r["pack_version"], "condition": r["condition"], "register": r["footing"],
               "scope": r["list"], "sample": r["sample"], "temporal": r["temporal"],
               "ratings": {x["option"]: x["score"] for x in r["ratings"]},
               "response": r["response"], "n_options": r["n_options"], "n_rated": r["n_rated"],
               "coverage": r["coverage"], "finish_reason": r["finish_reason"], "error_type": r["error_type"],
               "error": r["error"], "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
               "tokens_reasoning": r["tokens_reasoning"], "latency_ms": r["latency_ms"], "ts": r["ts"]}
        (main if r["source"] == "main" else probe).append(row)
    write_jsonl(store / "runs/rating_v1/all/ratings.jsonl", main)
    write_jsonl(store / "runs/qualify/probe.jsonl", probe)
    print(f"  rating_attempts {len(main):>7,d} main-run rows -> runs/rating_v1/all/ratings.jsonl")
    print(f"                  {len(probe):>7,d} probe rows -> runs/qualify/probe.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default="v1.0.0", help="dataset tag or commit (default v1.0.0)")
    ap.add_argument("--source", help="read a local build of the dataset instead of downloading")
    ap.add_argument("--runs", help="also rebuild the raw run store in this folder")
    ap.add_argument("--data", default=str(DATA), help="where to write the derived tables (default data/)")
    args = ap.parse_args()

    manifest = json.loads((DATA / "MANIFEST.json").read_text())["outputs"]
    ok = True
    for name in ("entities", "ratings"):
        dest = Path(args.data) / f"{name}.jsonl"
        digest = restore_table(name, args, dest)
        match = digest == manifest.get(f"{name}.jsonl")
        ok &= match
        print(f"  {name:13s} -> {dest}  {'matches MANIFEST' if match else 'DOES NOT MATCH MANIFEST'}")
    if args.runs:
        rebuild_runs(args, Path(args.runs).resolve())
    if not ok:
        sys.exit("a derived table does not match data/MANIFEST.json; check the revision")


if __name__ == "__main__":
    main()
