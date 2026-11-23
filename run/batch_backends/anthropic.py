"""Anthropic Message Batches client (standard library only).

Submit a set of message requests as one batch, poll until it ends, and fetch the results
keyed by custom_id. Batches are billed at half the live rate and run server-side.

Interface shared with the other backends:
    run_requests(specs, poll, log) -> ({custom_id: assistant_text}, meta)
where each spec is a provider-neutral request dict:
    {custom_id, model, system?, messages:[{role, content}], temperature, max_tokens, extra?}
meta carries the batch id, request counts, summed usage, and per-item telemetry
(tokens, stop reason, served model, response id) or the per-item error.

Endpoints: POST /v1/messages/batches -> GET /v1/messages/batches/{id} until
processing_status == "ended" -> GET results_url (JSONL keyed by custom_id).
"""
import json
import os
import time
import urllib.error
import urllib.request

BASE = "https://api.anthropic.com/v1/messages/batches"


def _headers():
    return {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"}


_RETRY_CODES = {429, 500, 502, 503, 529}


def _req(url, data=None, method="GET", timeout=300, retries=6):
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=_headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:500]
            except Exception:
                pass
            if any(m in body.lower() for m in ("insufficient balance", "insufficient_quota",
                    "exceeded_current_quota", "credit balance", "suspended", "recharge",
                    "out of credit", "quota_exceeded", "depleted", "prepayment credit",
                    "billing")):  # exhausted -> fail fast
                raise RuntimeError(f"anthropic quota/balance exhausted (HTTP {e.code}): {body[:200]}") from None
            if e.code in _RETRY_CODES and attempt < retries - 1:
                time.sleep(min(2 ** attempt, 30)); continue
            raise RuntimeError(f"anthropic {method} {url} -> {e.code}: {body[:200]}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 30)); continue
            raise


def _to_request(s):
    params = {"model": s["model"], "max_tokens": s["max_tokens"],
              "messages": [{"role": m["role"], "content": m["content"]} for m in s["messages"]]}
    if s.get("temperature") is not None:  # newest Claudes reject temperature
        params["temperature"] = s["temperature"]
    if s.get("system"):
        params["system"] = s["system"]
    params.update(s.get("extra", {}))
    return {"custom_id": s["custom_id"], "params": params}


def create_batch(specs):
    body = json.dumps({"requests": [_to_request(s) for s in specs]}).encode()
    return _req(BASE, data=body, method="POST")


def get_batch(batch_id):
    return _req(f"{BASE}/{batch_id}")


def _download(results_url, retries=6):
    for attempt in range(retries):
        req = urllib.request.Request(results_url, headers=_headers())
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read().decode()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 30)); continue
            raise


def run_requests(specs, poll=30, log=print):
    """Full round-trip. Returns ({custom_id: text}, meta). Blocks until the batch
    ends; run in the background for large jobs."""
    b = create_batch(specs)
    bid = b["id"]
    log(f"submitted anthropic batch {bid}: {len(specs)} requests")
    while True:
        b = get_batch(bid)
        log(f"  batch {bid}: {b['processing_status']} {b.get('request_counts', {})}")
        if b["processing_status"] == "ended":
            break
        time.sleep(poll)
    out, usage, stop, per_item = {}, {"input": 0, "cached": 0, "output": 0}, {}, {}
    for line in _download(b["results_url"]).splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid, res = rec["custom_id"], rec.get("result", {})
        if res.get("type") == "succeeded":
            msg = res["message"]
            out[cid] = "".join(blk.get("text", "") for blk in msg.get("content", [])
                               if blk.get("type") == "text")
            stop[cid] = msg.get("stop_reason")
            u = msg.get("usage", {}) or {}
            usage["input"] += u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
            usage["cached"] += u.get("cache_read_input_tokens", 0)
            usage["output"] += u.get("output_tokens", 0)
            per_item[cid] = {"tokens_in": u.get("input_tokens"), "tokens_out": u.get("output_tokens"),
                             "tokens_cached": u.get("cache_read_input_tokens"),
                             "finish_reason": msg.get("stop_reason"), "stop_sequence": msg.get("stop_sequence"),
                             "served_model": msg.get("model"), "response_id": msg.get("id")}
        else:
            log(f"  request {cid} {res.get('type')}: {str(res.get('error'))[:140]}")
            per_item[cid] = {"error": f"{res.get('type')}: {str(res.get('error'))[:120]}"}
    meta = {"batch_id": bid, "request_counts": b.get("request_counts"),
            "usage": usage, "stop_reasons": stop, "per_item": per_item}
    log(f"anthropic batch {bid} ended: {len(out)}/{len(specs)} results parsed")
    return out, meta
