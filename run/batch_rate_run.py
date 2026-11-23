"""Structured-rating arm via provider batch APIs: same grid and row schema as rate_run.py.

Each model's remaining cells (resume-aware against ratings.jsonl and text-bearing probe
rows) are submitted as one batch. Anthropic, OpenAI and Google only; other providers use
the live lanes.

Token budget: batch lanes ran at 8,192 output tokens (4,096 for gpt-3.5-turbo), whereas
the live lanes ran at 32,768 with a provider-driven fallback. Both budgets are recorded
per row via the finish reason; the difference should be stated wherever the budget is
reported.

    python3 run/batch_rate_run.py --models k1 k2 ... --out runs/rating_v1/<lane> [--limit N]
"""

import argparse
import builtins
import functools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

print = functools.partial(builtins.print, flush=True)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "run"))
import gen_run  # noqa: E402
import rate_run  # noqa: E402
import rating_instrument as rt  # noqa: E402
from batch_backends import anthropic as be_anthropic  # noqa: E402
from batch_backends import google as be_google  # noqa: E402
from batch_backends import openai as be_openai  # noqa: E402
from batch_gen_run import backend_for  # noqa: E402

MAXTOK = 8192


def cid(cond, footing, scope, samp):
    return f"{cond}-{footing}-{scope}-s{samp}"


def parse_cid(c):
    cond, footing, scope, s = c.rsplit("-", 3)
    return cond, footing, scope, int(s[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    models = gen_run.load_models(only=args.models)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ratings.jsonl"
    done = rate_run.load_done([out_path, rate_run.PROBE_PATH])

    jobs = []
    for mk, (provider, mid, base, env) in models.items():
        be = backend_for(provider, base)
        if not be:
            print(f"{mk}: no batch lane; use rate_run.py")
            continue
        specs = []
        for cond in rt.SIT:
            for footing in rate_run.FOOTINGS:
                for scope in rate_run.SCOPES:
                    for samp in range(rate_run.SAMPLES):
                        if (cond, footing, scope, mk, samp) in done:
                            continue
                        if args.limit and len(specs) >= args.limit:
                            break
                        msg, _ = rt.build_message(cond, footing, scope, rate_run.TEMPORAL,
                                                  f"{mk}|{cond}|{footing}|{scope}|{samp}")
                        specs.append({"custom_id": cid(cond, footing, scope, samp), "model": mid,
                                      "messages": [{"role": "user", "content": msg}],
                                      "temperature": None,
                                      "max_tokens": 4096 if mid == "gpt-3.5-turbo" else MAXTOK,
                                      "token_param": "max_completion_tokens" if mid.startswith(("gpt-5", "o")) else None})
        if specs:
            jobs.append((be, mk, mid, provider, specs))
    if not jobs:
        print("nothing to submit")
        return
    print(f"submitting {len(jobs)} rating batches: " + ", ".join(f"{mk}({len(s)})" for _, mk, _, _, s in jobs))

    def run(be, mk, specs):
        log = lambda s: print(f"  [{mk}] {s}")  # noqa: E731
        fn = {"anthropic": be_anthropic.run_requests, "openai": be_openai.run_requests,
              "google": be_google.run_requests}[be]
        return fn(specs, poll=args.poll, log=log)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool, out_path.open("a") as out_f:
        futs = {pool.submit(run, be, mk, specs): (be, mk, mid, provider, specs)
                for be, mk, mid, provider, specs in jobs}
        for fut in as_completed(futs):
            be, mk, mid, provider, specs = futs[fut]
            try:
                results, meta = fut.result()
            except Exception as e:
                print(f"  [{mk}] BATCH FAILED: {type(e).__name__}: {str(e)[:160]}")
                continue
            per_item = meta.get("per_item", {})
            n_ok = 0
            for s in specs:
                c = s["custom_id"]
                cond, footing, scope, samp = parse_cid(c)
                item = per_item.get(c, {})
                text = results.get(c)
                err = item.get("error") if text is None else None
                lmap = rt.name_lower_map(scope)
                parsed = rt.parse_ratings(text, lmap) if text else {}
                n_opts = len(lmap)
                row = {"arm": "rating", "condition": cond, "register": footing, "scope": scope,
                       "temporal": rate_run.TEMPORAL, "sample": samp, "model_key": mk,
                       "model_id": mid, "n_options": n_opts, "n_rated": len(parsed),
                       "coverage": len(parsed) / n_opts if n_opts else 0,
                       "ratings": parsed, "response": text, "latency_ms": None,
                       "error": err, "error_type": gen_run.classify_error(err) if err else None,
                       "pack_version": gen_run.PACK_VERSION, "ts": time.time(),
                       "param_style": f"batch:{be}",
                       **{k: v for k, v in item.items() if k != "error"},
                       "batch_id": meta.get("batch_id") or meta.get("batch_name")}
                out_f.write(json.dumps(row) + "\n")
                n_ok += err is None
            out_f.flush()
            print(f"  [{mk}] merged {len(specs)} rows ({n_ok} ok, batch {meta.get('batch_id') or meta.get('batch_name')})")


if __name__ == "__main__":
    main()
