"""Export a generation run into the sidecar files that downstream steps read: one
items-<model>.jsonl per model with rows {item_key, input_ref, output, scores, raw} and a
manifest {schema_version, parent_artifact, benchmark, model, n_items, sha256}. The judge
pass (run/batch_judge_run.py) and the analysis chain read these. parent_artifact is null
until the dataset is published.

Only usable rows are exported: no error, non-empty response, finish reason not a length
cap. Cells with several rows (re-runs on resume) keep the last usable one.

    python3 run/export_run.py --run runs/gen_v1 [--out <dir>]
Writes to <run>/export/sidecar/.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = ROOT / args.run
    # single-dir runs have generations.jsonl at the top; lane runs (run_lanes.py) have
    # one per account subdirectory — merge them
    sources = ([run_dir / "generations.jsonl"] if (run_dir / "generations.jsonl").exists()
               else sorted(run_dir.glob("*/generations.jsonl")))
    if not sources:
        raise SystemExit(f"no generations.jsonl under {run_dir}")
    rows, malformed = [], 0
    for src in sources:
        for l in src.open():
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                malformed += 1  # torn final line from a hard crash; the cell reruns on resume
    if len(sources) > 1:
        print(f"merged {len(sources)} lane files")
    if malformed:
        print(f"note: {malformed} malformed line(s) skipped (crash-torn; rerun the run command to fill)")
    out = Path(args.out) if args.out else run_dir / "export"
    def usable(r):
        return not r.get("error") and (r.get("response") or "").strip() \
            and str(r.get("finish_reason")).lower() not in ("length", "max_tokens")
    # dedupe by cell, keeping the last usable row (earlier attempts may be empty/truncated)
    by_cell = {}
    for r in rows:
        if usable(r):
            by_cell[(r["condition"], r["register"], r["category"], r["model_key"], r["sample"])] = r
    ok = list(by_cell.values())
    errs = Counter(r["model_key"] for r in rows if r.get("error"))
    unusable = len([r for r in rows if not r.get("error") and not usable(r)])
    if unusable:
        print(f"note: {unusable} empty/truncated rows excluded (cells re-run on resume)")

    # ---- platform sidecars (one per model, per contract) ----
    side = out / "sidecar"; side.mkdir(parents=True, exist_ok=True)
    by_model = {}
    for r in ok:
        by_model.setdefault(r["model_key"], []).append(r)
    for mk, mrows in sorted(by_model.items()):
        items = side / f"items-{mk}.jsonl"
        with items.open("w") as f:
            for r in sorted(mrows, key=lambda r: (r["condition"], r["register"], r["category"], r["sample"])):
                item = {
                    "item_key": f'{r["condition"]}|{r["register"]}|{r["category"]}|s{r["sample"]}',
                    "input_ref": f'mindbench:intervention-audit/prompt-pack@{r["pack_version"]}'
                                 f'#{r["condition"]}/{r["register"]}/{r["category"]}',
                    "output": r["response"],
                    "scores": {},  # populated by the extraction pass
                    # everything not already represented in item_key/output/input_ref rides
                    # along in raw, so nothing recorded at run time is lost
                    "raw": {k: v for k, v in r.items() if k not in
                            ("condition", "register", "category", "sample", "response", "pack_version")},
                }
                f.write(json.dumps(item) + "\n")
        manifest = {"schema_version": 0, "parent_artifact": None,
                    "benchmark": "intervention-audit-generation",
                    "model": mrows[0]["model_id"], "model_key": mk,
                    "pack_version": mrows[0]["pack_version"],
                    "n_items": len(mrows), "sha256": sha256(items)}
        (side / f"manifest-{mk}.json").write_text(json.dumps(manifest, indent=1))

    print(f"exported {len(ok)} rows -> {side.relative_to(ROOT) if side.is_relative_to(ROOT) else side} "
          f"({len(by_model)} sidecars)")
    if errs:
        print(f"note: {sum(errs.values())} error rows excluded: {dict(errs)}")


if __name__ == "__main__":
    main()
