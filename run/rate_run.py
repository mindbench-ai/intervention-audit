"""Structured-rating arm, live lanes: per-class isolation scopes over the full grid.

Grid per model: 19 presentations x 3 footings x 4 scopes x 3 samples = 684 cells. Same
instrument as the qualification probe (run/rating_instrument.py), same telemetry, resume
by cell. Probe rows that carry response text count as done cells, so each qualified
model's 24 probe cells are not re-run.

    python3 run/rate_run.py --models k1 k2 ... --out runs/rating_v1/<lane>

Batch-eligible providers ran through run/batch_rate_run.py instead.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "run"))
import anthropic  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

import gen_run  # noqa: E402
import rate_qualify as rq  # noqa: E402  (MAXTOK, SERIAL_MODELS)
import rating_instrument as rt  # noqa: E402

FOOTINGS = ["clinical", "caregiver", "patient"]
SCOPES = rt.CLASSES
SAMPLES = 3
TEMPORAL = "initial"
PROBE_PATH = ROOT / "runs/qualify/probe.jsonl"


def load_done(paths):
    """Cells settled by a non-error row with response text and at least one parsed rating."""
    done = set()
    for p in paths:
        if not Path(p).exists():
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cov = r.get("coverage", r.get("n_rated", 0) / max(r.get("n_options", 1), 1))
            if r.get("error") is None and r.get("response") and cov > 0:
                done.add((r["condition"], r.get("register", "clinical"), r["scope"], r["model_key"], r["sample"]))
    return done


async def run_model(mk, provider, mid, base, env, out_f, done, conc):
    client = (anthropic.AsyncAnthropic(api_key=os.environ[env]) if provider == "anthropic"
              else AsyncOpenAI(api_key=os.environ[env], base_url=base))
    sem = asyncio.Semaphore(conc)
    n_done = n_run = 0
    hard_errors = {"n": 0}  # consecutive non-retryable failures; a model failing identically stops early

    async def one(cond, footing, scope, samp):
        nonlocal n_run
        lmap = rt.name_lower_map(scope)
        msg, n_opts = rt.build_message(cond, footing, scope, TEMPORAL, f"{mk}|{cond}|{footing}|{scope}|{samp}")
        async with sem:
            if env in gen_run.BREAKER or mk in gen_run.BREAKER:
                return
            text, meta, err, t0 = None, {}, None, time.monotonic()
            for att in range(3):
                try:
                    text, meta = await rt.call(provider, client, mid, msg, rq.MAXTOK)
                    err = None
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    etype = gen_run.classify_error(err)  # includes the exception name for class-based patterns
                    if etype in ("out_of_credit", "auth"):
                        gen_run.BREAKER.setdefault(env, err)
                        print(f"  !! circuit OPEN for {env}: {err[:80]}")
                        return
                    if etype in gen_run.NON_RETRYABLE:
                        hard_errors["n"] += 1
                        if hard_errors["n"] >= 8 and mk not in gen_run.BREAKER:
                            gen_run.BREAKER[mk] = f"8+ consecutive non-retryable errors: {err[:80]}"
                            print(f"  !! circuit OPEN for model {mk}: identical hard failures")
                        break
                    await asyncio.sleep(2 ** att)
            if err is None:
                hard_errors["n"] = 0
            parsed = rt.parse_ratings(text, lmap) if text else {}
            row = {"arm": "rating", "condition": cond, "register": footing, "scope": scope,
                   "temporal": TEMPORAL, "sample": samp, "model_key": mk, "model_id": mid,
                   "n_options": n_opts, "n_rated": len(parsed),
                   "coverage": len(parsed) / n_opts if n_opts else 0,
                   "ratings": parsed, "response": text,
                   "latency_ms": round((time.monotonic() - t0) * 1000),
                   "error": err, "error_type": gen_run.classify_error(err) if err else None,
                   "pack_version": gen_run.PACK_VERSION, "ts": time.time(), **meta}
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            n_run += 1

    tasks = []
    for cond in rt.SIT:
        for footing in FOOTINGS:
            for scope in SCOPES:
                for samp in range(SAMPLES):
                    if (cond, footing, scope, mk, samp) in done:
                        n_done += 1
                        continue
                    tasks.append(one(cond, footing, scope, samp))
    print(f"  {mk}: {len(tasks)} cells to run ({n_done} already done)")
    await asyncio.gather(*tasks)
    print(f"  {mk}: finished ({n_run} new cells)")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=3)
    args = ap.parse_args()
    models = gen_run.load_models(only=args.models)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ratings.jsonl"
    done = load_done([out_path, PROBE_PATH])
    with out_path.open("a") as out_f:
        for mk, (provider, mid, base, env) in models.items():
            conc = 1 if mk in rq.SERIAL_MODELS else args.conc
            await run_model(mk, provider, mid, base, env, out_f, done, conc)


if __name__ == "__main__":
    asyncio.run(main())
