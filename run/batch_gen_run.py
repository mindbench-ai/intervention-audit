"""Open-generation arm via provider batch APIs: same grid and row schema as gen_run.py.

For each model on a provider with a batch lane (Anthropic Message Batches, OpenAI Batch
API, Google's native batch endpoint), every outstanding cell is packaged into one batch
job, submitted, polled to completion, and merged into <out>/generations.jsonl. Rows use
the live runner's schema with latency_ms = null and param_style = "batch:<backend>", so
resume, export, and analysis are shared: a cell completed by either runner counts as done
for both. Models on providers without a batch lane (xAI, DeepSeek, Moonshot, Mistral,
DashScope-served Qwen) are reported and left to gen_run.py.

Temperature is omitted from every request (provider default). Newer OpenAI models take
the token budget as max_completion_tokens; everything else as max_tokens.

    python3 run/batch_gen_run.py --models claude-opus-5 gpt-5.6-sol --out runs/gen_v1/batch

A whole-batch failure writes no rows; the cells are picked up on the next run.
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

print = functools.partial(builtins.print, flush=True)  # background logs must stream, not buffer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "run"))
import gen_run  # noqa: E402  (SIT, ASKS, PACK, compose_prompt, is_done, load_models, MAX_TOKENS, classify_error)
from batch_backends import anthropic as be_anthropic  # noqa: E402
from batch_backends import google as be_google  # noqa: E402
from batch_backends import openai as be_openai  # noqa: E402


def backend_for(provider, base_url):
    if provider == "anthropic":
        return "anthropic"
    if base_url and "api.openai.com" in base_url:
        return "openai"
    if base_url and "generativelanguage.googleapis.com" in base_url:
        return "google"
    return None


def cid(cond, footing, cat, samp):
    return f"{cond}-{footing}-{cat}-s{samp}"  # Anthropic custom_id charset is [A-Za-z0-9_-]


def parse_cid(c):
    cond, footing, cat, s = c.rsplit("-", 3)
    return cond, footing, cat, int(s[1:])


def done_cells(out_path):
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if gen_run.is_done(r):
                done.add((r["condition"], r["register"], r["category"], r["model_key"], r["sample"]))
    return done


def run_model(backend, mk, specs, poll):
    log = lambda s: print(f"  [{mk}] {s}")  # noqa: E731
    if backend == "anthropic":
        return be_anthropic.run_requests(specs, poll=poll, log=log)
    if backend == "openai":
        return be_openai.run_requests(specs, poll=poll, log=log)
    return be_google.run_requests(specs, poll=poll, log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all", choices=["core", "wide", "all"])
    ap.add_argument("--models", nargs="*", help="explicit model keys from config (overrides --tier)")
    ap.add_argument("--samples", type=int, default=gen_run.SAMPLES)
    ap.add_argument("--out", default="runs/gen_v1/batch", help="output directory")
    ap.add_argument("--limit", type=int, default=0, help="cap cells per model (smoke runs)")
    ap.add_argument("--poll", type=int, default=60, help="seconds between status polls")
    args = ap.parse_args()

    models = gen_run.load_models(tier=args.tier, only=args.models)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"
    done = done_cells(out_path)

    jobs, no_lane = [], set()
    for mk, (provider, mid, base, env) in models.items():
        be = backend_for(provider, base)
        if not be:
            no_lane.add(mk)
            continue
        specs = []
        for cond in gen_run.SIT:
            for footing in gen_run.FOOTINGS:
                for cat in gen_run.CATEGORIES:
                    for s in range(args.samples):
                        if (cond, footing, cat, mk, s) in done:
                            continue
                        if args.limit and len(specs) >= args.limit:
                            break
                        specs.append({"custom_id": cid(cond, footing, cat, s), "model": mid,
                                      "messages": [{"role": "user",
                                                    "content": gen_run.compose_prompt(cond, footing, cat)}],
                                      "temperature": None, "max_tokens": gen_run.MAX_TOKENS,
                                      "token_param": ("max_completion_tokens"
                                                      if mid.startswith(("gpt-5", "o")) else None)})
        if specs:
            jobs.append((be, mk, mid, provider, specs))
    if no_lane:
        print(f"no batch lane (run live via gen_run.py): {sorted(no_lane)}")
    if not jobs:
        print("nothing to submit")
        return
    print(f"submitting {len(jobs)} batches: " + ", ".join(f"{mk}({len(s)})" for _, mk, _, _, s in jobs))

    meta_path = out_dir / "run_metadata.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "pack_version": gen_run.PACK_VERSION, "pack_manifest": gen_run.PACK,
            "tier": args.tier, "explicit_models": args.models, "samples": args.samples,
            "max_tokens": gen_run.MAX_TOKENS, "temperature": "provider-default",
            "mode": "batch", "models": {mk: {"provider": p, "model_id": mid} for _, mk, mid, p, _ in jobs},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))

    n_ok = 0
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool, out_path.open("a") as out_f:
        futs = {pool.submit(run_model, be, mk, specs, args.poll): (be, mk, mid, provider, specs)
                for be, mk, mid, provider, specs in jobs}
        for fut in as_completed(futs):
            be, mk, mid, provider, specs = futs[fut]
            try:
                results, meta = fut.result()
            except Exception as e:
                print(f"  [{mk}] BATCH FAILED: {type(e).__name__}: {str(e)[:160]}")
                continue
            per_item = meta.get("per_item", {})
            for s in specs:
                c = s["custom_id"]
                cond, footing, cat, samp = parse_cid(c)
                item = per_item.get(c, {})
                text = results.get(c)
                err = item.get("error") if text is None else None
                row = {"condition": cond, "register": footing, "category": cat, "model_key": mk,
                       "model_id": mid, "provider": provider, "sample": samp,
                       "prompt": s["messages"][0]["content"], "response": text,
                       "empty_response": (err is None and not (text or "").strip()),
                       "error": err, "error_type": gen_run.classify_error(err) if err else None,
                       "attempts": 1, "retry_errors": [], "latency_ms": None,
                       "ts": time.time(), "pack_version": gen_run.PACK_VERSION,
                       "param_style": f"batch:{be}",
                       **{k: v for k, v in item.items() if k != "error"},
                       "batch_id": meta.get("batch_id") or meta.get("batch_name")}
                out_f.write(json.dumps(row) + "\n")
                n_ok += err is None
            out_f.flush()
            print(f"  [{mk}] merged {len(specs)} rows (batch {meta.get('batch_id') or meta.get('batch_name')})")
    print(f"finished: {n_ok} ok rows written")


if __name__ == "__main__":
    main()
