"""Open-generation arm: send every prompt in the grid to every model.

Prompt = vignette + " " + ask, both from the frozen instrument pack
(instrument/dataset/situations.json, instrument/dataset/asks.json). Sent as a single user
message with no system prompt; temperature and reasoning settings are the provider
defaults. Grid: 19 presentations x 3 footings x 5 asks x 3 samples = 855 calls per model.

One JSON row is appended per call to <out>/generations.jsonl with the response text and
the telemetry the provider returned (tokens, finish reason, served model name,
fingerprint, reasoning text where exposed). A run_metadata.json is written once per
output directory with the settings and the pack manifest.

Resumable: on restart, a cell counts as done only if its stored row has no error, a
non-empty response, and a finish reason other than a length cap; everything else is
re-queued. This is how length-capped cells were re-run at a higher budget
(--max-tokens 10000) after the main pass.

    python3 run/gen_run.py --models claude-opus-5 gpt-5.6-sol --out runs/gen_v1/live
    python3 run/gen_run.py --tier core --out runs/gen_v1/live --max-tokens 10000

Batch-eligible providers (Anthropic, OpenAI, Google) were run through run/batch_gen_run.py
instead, which writes rows of the same schema.
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
import anthropic  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

# Reasoning-default models spend their budget on hidden thinking first; the initial 800-token
# cap left some of them with empty visible responses, so the standard budget is 4,096 and
# length-capped cells were refilled at 10,000 (see --max-tokens).
MAX_TOKENS = 4096
RETRIES = 4
CONC = 6
SAMPLES = 3
FOOTINGS = ["clinical", "caregiver", "patient"]

INSTRUMENT = ROOT / "instrument"
SIT = {k: v for k, v in json.loads((INSTRUMENT / "dataset/situations.json").read_text()).items()
       if not k.startswith("_")}
ASKS = {k: v for k, v in json.loads((INSTRUMENT / "dataset/asks.json").read_text()).items()
        if not k.startswith("_")}
PACK = json.loads((INSTRUMENT / "VERSION.json").read_text())
PACK_VERSION = PACK["version"]
CATEGORIES = list(ASKS.keys())  # open, medication, psychotherapy, neuromodulation, digital

MODELS = {}  # key -> (provider, model_id, base_url, env_key); filled in main()


def load_models(tier="all", only=None):
    """Roster from config/models.json -> {key: (provider, model_id, base_url, env_key)}."""
    cfg = json.loads((ROOT / "config/models.json").read_text())
    out = {}
    for k, m in cfg["models"].items():
        if only and k not in only:
            continue
        if tier != "all" and m["tier"] != tier and not only:
            continue
        out[k] = (m["provider"], m["model_id"], m["base_url"], m["env_key"])
    return out


def compose_prompt(cond, footing, cat):
    return f"{SIT[cond][footing]} {ASKS[cat][footing]}"


def _g(obj, *path):
    for p in path:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


async def call(provider, client, mid, msg):
    """Returns (text, meta). meta captures what the response object exposes: usage
    (including reasoning and cached token detail where given), finish reason, refusal
    signal, the provider's echo of the served model (the pinning record for floating-alias
    providers), the system fingerprint, and the request-parameter style used after any
    fallback."""
    if provider == "anthropic":
        r = await client.messages.create(model=mid, max_tokens=MAX_TOKENS,
                                         messages=[{"role": "user", "content": msg}])
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, {"tokens_in": r.usage.input_tokens, "tokens_out": r.usage.output_tokens,
                      "tokens_cached": _g(r.usage, "cache_read_input_tokens"),
                      "finish_reason": r.stop_reason, "stop_sequence": r.stop_sequence,
                      "served_model": r.model, "response_id": r.id,
                      "param_style": "max_tokens"}
    param_style = "max_tokens"
    try:
        r = await client.chat.completions.create(model=mid, max_tokens=MAX_TOKENS,
                                                 messages=[{"role": "user", "content": msg}])
    except Exception as e:
        if "max_completion_tokens" in str(e):
            # newer OpenAI models reject max_tokens in favor of max_completion_tokens
            param_style = "max_completion_tokens"
            r = await client.chat.completions.create(model=mid, max_completion_tokens=MAX_TOKENS,
                                                     messages=[{"role": "user", "content": msg}])
        elif "enable_thinking" in str(e):
            # DashScope-served Qwen3 open models need thinking disabled for non-streaming calls
            param_style = "max_tokens+enable_thinking=false"
            r = await client.chat.completions.create(model=mid, max_tokens=MAX_TOKENS,
                                                     messages=[{"role": "user", "content": msg}],
                                                     extra_body={"enable_thinking": False})
        else:
            raise
    ch = r.choices[0]
    return (ch.message.content or ""), {
        "reasoning_content": _g(ch, "message", "reasoning_content"),  # DeepSeek/Qwen/Kimi expose it
        "tokens_in": _g(r, "usage", "prompt_tokens"), "tokens_out": _g(r, "usage", "completion_tokens"),
        "tokens_reasoning": _g(r, "usage", "completion_tokens_details", "reasoning_tokens"),
        "tokens_cached": _g(r, "usage", "prompt_tokens_details", "cached_tokens"),
        "finish_reason": ch.finish_reason, "api_refusal": _g(ch, "message", "refusal"),
        "served_model": r.model, "response_id": _g(r, "id"), "server_created": _g(r, "created"),
        "system_fingerprint": _g(r, "system_fingerprint"), "param_style": param_style}


def classify_error(err):
    e = err.lower()
    # Billing and quota exhaustion first: these arrive as 429s and 400s whose class looks
    # transient (an OpenAI insufficient_quota is a RateLimitError), so the message text
    # has to be matched before the exception-name patterns.
    for pat in ("insufficient_quota", "insufficient balance", "credit balance", "payment required",
                "exceeded your current quota", "billing", "arrears", "top up", "top-up"):
        if pat in e:
            return "out_of_credit"
    for pat, label in (("ratelimit", "rate_limit"), ("timeout", "timeout"), ("connection", "connection"),
                       ("authentication", "auth"), ("permission", "auth"), ("badrequest", "bad_request"),
                       ("notfound", "bad_request"), ("invalid_parameter", "bad_request"),
                       ("invalidparameter", "bad_request"), ("internalserver", "server"),
                       ("overloaded", "server"), ("apistatus", "server")):
        if pat in e:
            return label
    return "other"


# Account-level circuit breaker: once a key reports out-of-credit or dead auth, every
# remaining queued call on that key is skipped without touching the API. Skipped cells are
# not written, so a resumed run picks them up cleanly.
BREAKER = {}  # env_key or model_key -> first fatal error
NON_RETRYABLE = ("out_of_credit", "auth", "bad_request")
SKIPPED = {"n": 0}
EMPTY_STREAK = {}  # model_key -> consecutive empty responses
EMPTY_STREAK_LIMIT = 8  # a model this consistently empty is misconfigured; its lane stops calling it


async def run_one(sem, lock, out_f, clients, mk, cond, footing, cat, samp):
    provider, mid, _, env = MODELS[mk]
    msg = compose_prompt(cond, footing, cat)
    async with sem:
        if env in BREAKER or mk in BREAKER:
            SKIPPED["n"] += 1
            return False
        text, meta, err, latency_ms, retry_errors = None, {}, None, None, []
        for att in range(RETRIES + 1):
            t0 = time.monotonic()
            try:
                text, meta = await call(provider, clients[mk], mid, msg)
                latency_ms = round((time.monotonic() - t0) * 1000)
                err = None
                break
            except Exception as e:
                err_full = f"{type(e).__name__}: {str(e)}"
                err = err_full[:300]
                retry_errors.append(err)
                # classify on the full message: truncation once hid 'insufficient balance' in a 429
                etype = classify_error(err_full)
                if etype in ("out_of_credit", "auth"):
                    BREAKER.setdefault(env, err)
                    print(f"  !! circuit OPEN for {env} ({etype}): {err[:80]}")
                    break
                if etype in NON_RETRYABLE:
                    break
                if att < RETRIES:
                    await asyncio.sleep((2 ** att) + random.random())
        empty = (err is None and not (text or "").strip())
        if err is None:
            EMPTY_STREAK[mk] = EMPTY_STREAK.get(mk, 0) + 1 if empty else 0
            if EMPTY_STREAK[mk] >= EMPTY_STREAK_LIMIT and mk not in BREAKER:
                BREAKER[mk] = f"{EMPTY_STREAK[mk]} consecutive empty responses (model-level circuit)"
                print(f"  !! circuit OPEN for model {mk}: {BREAKER[mk]}")
        row = {"condition": cond, "register": footing, "category": cat, "model_key": mk,
               "model_id": mid, "provider": provider, "sample": samp, "prompt": msg,
               "response": text, "empty_response": empty,
               "error": err, "error_type": classify_error(err) if err else None,
               "attempts": att + 1, "retry_errors": retry_errors[:-1] if err else retry_errors,
               "latency_ms": latency_ms, "ts": time.time(),
               "pack_version": PACK_VERSION, **meta}
        async with lock:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
        return err is None


def is_done(row):
    """A stored row counts as done only if it produced usable text."""
    return (row.get("error") is None and (row.get("response") or "").strip()
            and str(row.get("finish_reason")).lower() not in ("length", "max_tokens"))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap total calls (smoke runs); 0 = full grid")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--out", default="runs/gen_v1/live", help="output directory")
    ap.add_argument("--tier", default="all", choices=["core", "wide", "all"])
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override MAX_TOKENS (used for the 10,000-token refill of length-capped cells)")
    ap.add_argument("--conc", type=int, default=None,
                    help="override concurrency (rate-limited providers were refilled at --conc 1 or 2)")
    ap.add_argument("--models", nargs="*", help="explicit model keys from config (overrides --tier)")
    args = ap.parse_args()
    global MAX_TOKENS, CONC
    if args.max_tokens:
        MAX_TOKENS = args.max_tokens
    if args.conc:
        CONC = args.conc
    MODELS.update(load_models(tier=args.tier, only=args.models))
    missing = [k for k, (_, _, _, env) in MODELS.items() if not os.environ.get(env)]
    if missing:
        raise SystemExit(f"missing API keys for: {missing}")
    print(f"{len(MODELS)} models loaded from config (tier={args.tier})")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "run_metadata.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "pack_version": PACK_VERSION, "pack_manifest": PACK,
            "tier": args.tier, "explicit_models": args.models, "samples": args.samples,
            "max_tokens": MAX_TOKENS, "temperature": "provider-default",
            "retries": RETRIES, "concurrency": CONC,
            "param_fallbacks": "max_completion_tokens (newer OpenAI); enable_thinking=false "
                               "(DashScope Qwen3); recorded per row as param_style",
            "models": {k: {"provider": v[0], "model_id": v[1]} for k, v in MODELS.items()},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))

    out_path = out_dir / "generations.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_done(r):
                done.add((r["condition"], r["register"], r["category"], r["model_key"], r["sample"]))

    clients = {}
    for mk, (prov, _, base, env) in MODELS.items():
        clients[mk] = (anthropic.AsyncAnthropic(api_key=os.environ[env]) if prov == "anthropic"
                       else AsyncOpenAI(api_key=os.environ[env], base_url=base))
    sem, lock, tasks = asyncio.Semaphore(CONC), asyncio.Lock(), []
    with out_path.open("a") as out_f:
        for cond in SIT:
            for footing in FOOTINGS:
                for cat in CATEGORIES:
                    for s in range(args.samples):
                        for mk in MODELS:  # innermost so capped smoke runs touch every provider
                            if (cond, footing, cat, mk, s) in done:
                                continue
                            if args.limit and len(tasks) >= args.limit:
                                break
                            tasks.append(run_one(sem, lock, out_f, clients, mk, cond, footing, cat, s))
        print(f"{len(tasks)} generation calls ({len(done)} done)")
        ok = 0
        for i, c in enumerate(asyncio.as_completed(tasks), 1):
            ok += bool(await c)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} ({ok} ok)")
    if BREAKER:
        print(f"circuit-broken accounts (skipped {SKIPPED['n']} calls, unwritten; resume after top-up):")
        for k, err in BREAKER.items():
            print(f"  {k}: {err[:100]}")
    print("finished")


if __name__ == "__main__":
    asyncio.run(main())
