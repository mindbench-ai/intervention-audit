"""Judge pass over the full generation corpus: extract named interventions and stances.

Reads the usable generation rows from the sidecar exports of both generation runs
(runs/gen_v1 and runs/legacy_v1, written by run/export_run.py: 41 models x 855 cells =
35,055 responses), sends each response after the frozen judge prompt
(run/judge_prompt.py; sha256 stamped on every row) to the judge model through the OpenAI
Batch API, and writes <out>/judgments.jsonl.

Robustness:
- Requests are chunked under the Batch API size and count limits and submitted one
  chunk at a time.
- Submitted-but-uncollected batch ids persist in batches.json; on restart those are
  collected first, so a crash between submit and collect never double-spends.
- Resume by custom_id: a cell is done only when its annotation parsed; parse failures and
  request errors are recorded and resubmitted on the next run.
- --live judges any cells still outstanding with live calls under the same prompt and
  budget, marking the rows judge = "luna-live". In the audited run one cell failed to
  parse twice in batch and was completed this way.

    python3 run/batch_judge_run.py [--limit N] [--out runs/judge_v1/luna] [--live]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "run"))
from batch_backends import openai as ob  # noqa: E402
from judge_prompt import PROMPT, PROMPT_SHA256  # noqa: E402

MODEL_ID = "gpt-5.6-luna"
MAX_TOK = 8000
CHUNK_BYTES = 120 * 1024 * 1024
CHUNK_REQS = 20000
PACK_VERSION = json.loads((ROOT / "instrument/VERSION.json").read_text())["version"]
# list price per million tokens (input, cached input, output); the batch discount is applied below
PRICE = (0.2, 0.02, 1.2)


def load_corpus():
    rows = {}
    for arm in ("gen_v1", "legacy_v1"):
        for f in sorted((ROOT / f"runs/{arm}/export/sidecar").glob("items-*.jsonl")):
            for line in f.open():
                r = json.loads(line)
                mk = r["raw"]["model_key"]
                rows[f"{mk}|{r['item_key']}"] = {"model_key": mk, "item_key": r["item_key"],
                                                 "arm": arm, "output": r["output"]}
    return rows


def load_done(out_path):
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("annotation") is not None and not r.get("error"):
                done.add(r["custom_id"])
    return done


def parse_annotation(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, "no JSON object in response"
    try:
        ann = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    if not isinstance(ann.get("entities"), list):
        return None, "missing entities list"
    return ann, None


def make_row(cid, src, text, telemetry, judge="luna-batch", error=None):
    cond, reg, cat, samp = (src.get("item_key", "|||").split("|") + [None] * 4)[:4]
    ann, perr = (None, error) if error else parse_annotation(text)
    return {"judge": judge, "judge_model": MODEL_ID, "custom_id": cid,
            "model_key": src.get("model_key"), "item_key": src.get("item_key"),
            "condition": cond, "register": reg, "category": cat, "sample": samp,
            "arm": src.get("arm"), "annotation": ann, "error": perr,
            "judge_raw": None if ann else text,
            "prompt_sha256": PROMPT_SHA256, "pack_version": PACK_VERSION,
            "ts": time.time(), **telemetry}


def write_results(out_f, corpus, results, per_item):
    n_ok = n_err = 0
    for cid, pi in per_item.items():
        row = make_row(cid, corpus.get(cid, {}), results.get(cid),
                       {k: v for k, v in pi.items() if k != "error"}, error=pi.get("error"))
        out_f.write(json.dumps(row) + "\n")
        n_ok += row["annotation"] is not None
        n_err += row["annotation"] is None
    out_f.flush()
    return n_ok, n_err


def collect(out_f, corpus, bid):
    b = ob.get_batch(bid)
    if b["status"] != "completed":
        return b["status"], set()
    results, usage, _stop, per_item = ob._parse_output(ob.download_file(b["output_file_id"]))
    ok, err = write_results(out_f, corpus, results, per_item)
    print(f"  collected {ok} ok / {err} err; usage {usage}")
    return "completed", set(per_item)


def judge_live(out_f, corpus, cids):
    """Live fallback for cells that keep failing in batch: same prompt, same budget."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for cid in cids:
        r = client.chat.completions.create(model=MODEL_ID, max_completion_tokens=MAX_TOK,
                                           messages=[{"role": "user", "content": PROMPT + corpus[cid]["output"]}])
        ch = r.choices[0]
        u = r.usage
        telemetry = {"tokens_in": u.prompt_tokens, "tokens_out": u.completion_tokens,
                     "finish_reason": ch.finish_reason, "served_model": r.model, "response_id": r.id}
        row = make_row(cid, corpus[cid], ch.message.content or "", telemetry, judge="luna-live")
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        print(f"  live: {cid} -> {'ok' if row['annotation'] is not None else row['error']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke: only submit the first N cells")
    ap.add_argument("--out", default="runs/judge_v1/luna")
    ap.add_argument("--live", action="store_true", help="judge outstanding cells with live calls instead of a batch")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "judgments.jsonl"
    state_path = out_dir / "batches.json"

    corpus = load_corpus()
    done = load_done(out_path)
    print(f"corpus {len(corpus)} cells, {len(done)} already judged, prompt {PROMPT_SHA256[:12]}")
    out_f = out_path.open("a")

    # collect any batches a previous run submitted but never collected
    state = json.loads(state_path.read_text()) if state_path.exists() else {"pending": []}
    for bid in list(state["pending"]):
        status, _ = collect(out_f, corpus, bid)
        print(f"reconciling pending batch {bid}: {status}")
        if status in ("failed", "expired", "cancelled"):
            print("  its cells will be resubmitted")
        elif status != "completed":
            print("  still in flight; waiting")
            ob.wait_batch(bid, poll=60)
            collect(out_f, corpus, bid)
    state["pending"] = []
    state_path.write_text(json.dumps(state))
    done = load_done(out_path)  # recount: only parseable annotations count as done

    todo = [cid for cid in corpus if cid not in done]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do")
        return
    if args.live:
        print(f"judging {len(todo)} cells live")
        judge_live(out_f, corpus, todo)
        print(f"TOTAL judged: {len(load_done(out_path))}/{len(corpus)}")
        return
    print(f"submitting {len(todo)} cells")

    chunks, cur, cur_bytes = [], [], 0
    for cid in todo:
        line = ob._to_line({"custom_id": cid, "model": MODEL_ID, "max_tokens": MAX_TOK,
                            "token_param": "max_completion_tokens", "temperature": None,
                            "messages": [{"role": "user", "content": PROMPT + corpus[cid]["output"]}]})
        nb = len(json.dumps(line)) + 1
        if cur and (cur_bytes + nb > CHUNK_BYTES or len(cur) >= CHUNK_REQS):
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(line)
        cur_bytes += nb
    chunks.append(cur)
    print(f"{len(chunks)} chunk(s): " + ", ".join(str(len(c)) for c in chunks))

    for ci, lines in enumerate(chunks):
        fid = ob.upload_jsonl(lines)
        bid = ob.create_batch(fid)
        state["pending"].append(bid)
        state_path.write_text(json.dumps(state))
        print(f"chunk {ci + 1}/{len(chunks)}: batch {bid} submitted ({len(lines)} requests)")
        b = ob.wait_batch(bid, poll=120)
        if b["status"] != "completed":
            print(f"  !! batch ended '{b['status']}': {str(b.get('errors'))[:200]}; re-run to resubmit its cells")
            state["pending"].remove(bid)
            state_path.write_text(json.dumps(state))
            continue
        results, usage, _stop, per_item = ob._parse_output(ob.download_file(b["output_file_id"]))
        ok, err = write_results(out_f, corpus, results, per_item)
        state["pending"].remove(bid)
        state_path.write_text(json.dumps(state))
        cost = (usage["input"] * PRICE[0] + usage["cached"] * PRICE[1] + usage["output"] * PRICE[2]) / 1e6
        print(f"  chunk {ci + 1} done: {ok} ok / {err} err; ~${cost * 0.5:.2f} at the batch discount")

    print(f"TOTAL judged: {len(load_done(out_path))}/{len(corpus)}")


if __name__ == "__main__":
    main()
