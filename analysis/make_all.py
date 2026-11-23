"""Run the analysis chain in order and write data/MANIFEST.json (sha256 of inputs and
outputs). With --check, rebuild into a temporary directory and report which outputs differ
from the committed ones.

    python3 analysis/make_all.py [--runs <folder with runs/>] [--check]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEPS = ["canonicalise.py", "ratings.py", "corpus.py", "cells.py", "disclosures.py", "mentions.py", "concordance.py", "models.py", "highlights.py", "conditioned.py", "brands.py", "authorities.py"]
OUTPUTS = ["entities.jsonl", "responses.jsonl", "ratings.jsonl", "rating_gaps.json", "corpus.json", "cells.json", "disclosures.json",
           "mentions.json", "concordance.json", "models.json", "highlights.json", "conditioned.json", "brands.json", "authorities.json"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(runs_dir, data_dir):
    env = {**os.environ, "MB_RUNS": str(runs_dir), "MB_DATA": str(data_dir)}
    for s in STEPS:
        print(f"== {s}")
        subprocess.run([sys.executable, str(HERE / s)], check=True, env=env, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", "--dev", dest="runs",
                    default=os.environ.get("MB_RUNS") or os.environ.get("MB_DEV") or str(ROOT))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    runs_dir = Path(args.runs).resolve()   # the steps run from analysis/, so pass an absolute path
    inputs = {str(p.relative_to(runs_dir)): sha(p) for p in
              [runs_dir / "runs/judge_v1/luna/judgments.jsonl", runs_dir / "runs/qualify/probe.jsonl"]
              + sorted((runs_dir / "runs/rating_v1").glob("*/ratings.jsonl")) if p.exists()}
    if args.check:
        tmp = Path(tempfile.mkdtemp())
        run(runs_dir, tmp)
        manifest = json.loads((ROOT / "data/MANIFEST.json").read_text())
        for o in OUTPUTS:
            same = sha(tmp / o) == manifest["outputs"].get(o)
            print(f"  {'ok ' if same else 'DIFF'} {o}")
        shutil.rmtree(tmp)
        return
    run(runs_dir, ROOT / "data")
    manifest = {"inputs": inputs, "outputs": {o: sha(ROOT / "data" / o) for o in OUTPUTS},
                "instrument": json.loads((ROOT / "instrument/VERSION.json").read_text())["files"]}
    (ROOT / "data/MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote data/MANIFEST.json")


if __name__ == "__main__":
    main()
