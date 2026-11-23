"""Per-account lane launcher for the live generation runner.

Groups the selected models by API account (one lane per key) and starts one run/gen_run.py
subprocess per lane, each writing to <out>/<lane>/generations.jsonl. Lanes are
independent: an account that runs out of credit trips only its own circuit breaker and
its lane ends while the others keep running. Re-running the same command after a top-up
resumes only the unfinished lanes (finished lanes exit at once with 0 calls).

    python3 run/run_lanes.py --tier core --out runs/gen_v1
    python3 run/run_lanes.py --models deepseek-v4-pro kimi-k3 --out runs/gen_v1

The audited live runs were launched this way; the batch-eligible providers ran through
run/batch_gen_run.py instead.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
import gen_run  # noqa: E402  (load_models, SAMPLES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all", choices=["core", "wide", "all"])
    ap.add_argument("--models", nargs="*", help="explicit model keys from config (overrides --tier)")
    ap.add_argument("--samples", type=int, default=gen_run.SAMPLES)
    ap.add_argument("--limit", type=int, default=0, help="per-lane call cap (smoke runs)")
    ap.add_argument("--out", required=True, help="parent output directory; one subdirectory per lane")
    args = ap.parse_args()

    models = gen_run.load_models(tier=args.tier, only=args.models)
    lanes = {}
    for mk, (_, _, _, env) in models.items():
        lanes.setdefault(env.replace("_API_KEY", "").lower(), []).append(mk)

    print(f"{len(models)} models across {len(lanes)} account lanes:")
    procs = {}
    for lane, keys in sorted(lanes.items()):
        lane_dir = ROOT / args.out / lane
        lane_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(ROOT / "run" / "gen_run.py"), "--models", *keys,
               "--samples", str(args.samples), "--out", f"{args.out}/{lane}"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        log = (lane_dir / "lane.log").open("a")
        procs[lane] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
        print(f"  {lane}: {len(keys)} models -> {args.out}/{lane}/ (pid {procs[lane][0].pid})")

    while procs:
        for lane, (p, log) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            log.close()
            tail = (ROOT / args.out / lane / "lane.log").read_text().splitlines()
            circuit = [line for line in tail if "circuit OPEN" in line]
            status = "CIRCUIT-BROKEN" if circuit else ("ok" if rc == 0 else f"exit {rc}")
            print(f"lane {lane} finished: {status}" + (f" | {circuit[-1].strip()}" if circuit else ""))
            del procs[lane]
        time.sleep(5)

    print("\nall lanes done. Per-lane state is independent; re-run this exact command "
          "after any top-up and finished lanes will no-op.")


if __name__ == "__main__":
    main()
