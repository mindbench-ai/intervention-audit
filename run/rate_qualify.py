"""Rating-arm qualification probe: format compliance before the full run.

For every core-tier model: 2 presentations (major depressive disorder and the bereavement
control) x the clinical footing x 4 class scopes x 3 samples = 24 calls. The probe checks
that a model returns parseable ratings for the option list; it is blind to what the
ratings say. Scoring against the gate (>= 80% of calls covering >= 90% of the options) is
done by validation/qualify_report.py; this file only collects.

Probe rows carry the full response text and count as done cells for the main run
(run/rate_run.py), so the 24 probe cells per model are part of the rating dataset.

    python3 run/rate_qualify.py [--models k1 k2 ...] [--out runs/qualify]
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

import gen_run  # noqa: E402  (load_models, classify_error, NON_RETRYABLE, PACK_VERSION)
import rating_instrument as rt  # noqa: E402

PROBE_CONDITIONS = ["major_depressive_disorder", "uncomplicated_bereavement"]
PROBE_FOOTING = "clinical"
SCOPES = rt.CLASSES
SAMPLES = 3
# Run-setting change (2026-08-23): the first budget of 8,192 left reasoning-default models
# with empty, length-capped responses. Compliance must not be confounded by budget, so the
# probe runs at 32,768; models already qualified at 8,192 are unaffected.
MAXTOK = 32768
RETRIES = 2  # the gate measures first-intent compliance; failures are not papered over with retries
SERIAL_MODELS = {"mistral-large-3"}  # provider rate-limits this model hard; one call at a time


async def probe_model(mk, provider, mid, base, env, out_f, cells):
    conc = 1 if mk in SERIAL_MODELS else 3
    client = (anthropic.AsyncAnthropic(api_key=os.environ[env]) if provider == "anthropic"
              else AsyncOpenAI(api_key=os.environ[env], base_url=base))
    sem = asyncio.Semaphore(conc)

    async def one(cond, scope, samp):
        lmap = rt.name_lower_map(scope)
        msg, n_opts = rt.build_message(cond, PROBE_FOOTING, scope, "initial", f"{mk}|{cond}|{scope}|{samp}")
        async with sem:
            text, meta, err, t0 = None, {}, None, time.monotonic()
            for att in range(RETRIES + 1):
                try:
                    text, meta = await rt.call(provider, client, mid, msg, MAXTOK)
                    err = None
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    if gen_run.classify_error(str(e)) in gen_run.NON_RETRYABLE:
                        break
                    await asyncio.sleep(2 ** att)
            parsed = rt.parse_ratings(text, lmap) if text else {}
            row = {"probe": "qualification", "condition": cond, "register": PROBE_FOOTING,
                   "scope": scope, "sample": samp, "model_key": mk, "model_id": mid,
                   "n_options": n_opts, "n_rated": len(parsed), "coverage": len(parsed) / n_opts,
                   "ratings": parsed, "response": text, "response_chars": len(text or ""),
                   "latency_ms": round((time.monotonic() - t0) * 1000), "error": err,
                   "error_type": gen_run.classify_error(err) if err else None,
                   "pack_version": gen_run.PACK_VERSION, "ts": time.time(), **meta}
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            return row

    rows = await asyncio.gather(*[one(c, s, i) for (c, s, i) in cells])
    cov = sorted(r["coverage"] for r in rows if not r["error"])
    med = cov[len(cov) // 2] if cov else 0
    print(f"  {mk}: {sum(1 for r in rows if not r['error'])}/{len(rows)} calls ok, median coverage {med:.0%}")


def starved(r):
    """An empty response that hit the length cap: an instrument artifact, not data."""
    return (not (r.get("response") or "").strip()) and str(r.get("finish_reason")).lower() in ("length", "max_tokens")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="explicit model keys from config (default: core tier)")
    ap.add_argument("--out", default="runs/qualify")
    args = ap.parse_args()
    models = gen_run.load_models(tier="core", only=args.models)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe.jsonl"
    done_cells = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None and not starved(r):
                done_cells.add((r["model_key"], r["condition"], r["scope"], r["sample"]))
    with out_path.open("a") as out_f:
        for mk, (provider, mid, base, env) in models.items():
            missing = [(c, sc, i) for c in PROBE_CONDITIONS for sc in SCOPES for i in range(SAMPLES)
                       if (mk, c, sc, i) not in done_cells]
            if not missing:
                continue
            print(f"  {mk}: probing {len(missing)} cells")
            try:
                await probe_model(mk, provider, mid, base, env, out_f, missing)
            except Exception as e:
                print(f"  {mk}: PROBE FAILED {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
