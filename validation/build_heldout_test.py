"""Held-out test of the frozen judge prompt: fresh responses annotated by the judge, for
human review of the prompt before it was frozen for the corpus pass.

Samples corpus responses never used in judge development (seed 23), one slot per ask x
footing with avoid-bearing and table-bearing responses salted in, annotates each with the
frozen prompt (run/judge_prompt.py) by live judge calls, and writes:
  - test20_luna<suffix>.jsonl   response + annotation per row
  - test20_viewer<suffix>.html  stance-colored highlights for reading

Two facts about the audited runs: the sampler asks for 20 but produced 19, because one
slot filters on a condition name that is not in the pack ("adjustment_disorder"; the
control is "situational_stress_subclinical") and never fills; and the pool is the main
generation run only, so the six legacy models were not in it. Three rounds were run
(validation/test20_luna_r1.jsonl is round 1, also used by the main-wave sampler as an
exclusion list). Each run makes about 20 live judge calls (roughly $0.10).

    python3 validation/build_heldout_test.py [--suffix _r4]
"""

import html
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
load_dotenv(ROOT / ".env")
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "run"))
from judge_prompt import PROMPT  # noqa: E402

MODEL_ID = "gpt-5.6-luna"
N = 20


def rid(r):
    return f"{r['model_key']}|{r['condition']}|{r['register']}|{r['category']}|s{r['sample']}"


def load_corpus():
    rows = []
    for p in sorted((ROOT / "runs/gen_v1").glob("*/generations.jsonl")):
        for l in p.open():
            if not l.strip().endswith("}"):
                continue
            r = json.loads(l)
            if not r.get("error") and (r.get("response") or "").strip() \
                    and str(r.get("finish_reason")).lower() not in ("length", "max_tokens"):
                rows.append(r)
    return rows


def sample20(rows, exclude):
    rng = random.Random(23)
    rng.shuffle(rows)
    picked, used_cond, used_model = [], set(), {}

    def take(pred, relax=False):
        for r in rows:
            if rid(r) in exclude or any(rid(r) == rid(p) for p in picked):
                continue
            # spread: new condition if possible, no model more than twice
            if not relax and r["condition"] in used_cond:
                continue
            if used_model.get(r["model_key"], 0) >= 2:
                continue
            if pred(r):
                picked.append(r)
                used_cond.add(r["condition"])
                used_model[r["model_key"]] = used_model.get(r["model_key"], 0) + 1
                return True
        return False

    lo = (lambda r: (r["response"] or "").lower())
    slots = [
        lambda r: r["category"] == "medication" and r["register"] == "clinical" and "avoid" in lo(r),
        lambda r: r["category"] == "medication" and r["register"] == "patient",
        lambda r: r["category"] == "medication" and r["register"] == "caregiver",
        lambda r: r["category"] == "medication" and "|" in (r["response"] or ""),
        lambda r: r["category"] == "medication" and r["register"] == "clinical",
        lambda r: r["category"] == "psychotherapy" and r["register"] == "patient",
        lambda r: r["category"] == "psychotherapy" and r["register"] == "clinical",
        lambda r: r["category"] == "psychotherapy" and r["register"] == "caregiver",
        lambda r: r["category"] == "neuromodulation" and r["register"] == "clinical",
        lambda r: r["category"] == "neuromodulation" and r["register"] == "patient",
        lambda r: r["category"] == "neuromodulation" and r["register"] == "caregiver",
        lambda r: r["category"] == "digital" and r["register"] == "patient",
        lambda r: r["category"] == "digital" and r["register"] == "clinical",
        lambda r: r["category"] == "open" and r["register"] == "patient",
        lambda r: r["category"] == "open" and r["register"] == "caregiver",
        lambda r: r["category"] == "open" and r["register"] == "clinical",
        lambda r: r["condition"] == "uncomplicated_bereavement",
        lambda r: r["condition"] == "adjustment_disorder",
        lambda r: r["category"] == "open" and "avoid" in lo(r),
        lambda r: True,
    ]
    for s in slots:
        take(s) or take(s, relax=True)
    return picked[:N]


def annotate(client, text):
    last = None
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL_ID, max_completion_tokens=8000,
                messages=[{"role": "user", "content": PROMPT + text}])
            raw = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", raw, re.S)
            return json.loads(m.group(0)), {"tin": r.usage.prompt_tokens, "tout": r.usage.completion_tokens}
        except Exception as e:
            last = e
    raise last


STANCE_BG = {"recommended": "#c9f0d0", "neutral": "#e2e8f0",
             "advised_against": "#fecaca", "mixed": "#e9d5ff"}


def render(text, ents):
    """Escape + highlight every occurrence of each reported surface (longest first)."""
    spans, taken = [], []
    for e in sorted(ents, key=lambda x: -len(x.get("surface") or "")):
        surf = (e.get("surface") or "").strip()
        if not surf:
            continue
        cs = surf.isupper() and len(surf) <= 5
        pat = re.compile(rf"(?<![\w-]){re.escape(surf)}(?![\w-])", 0 if cs else re.I)
        hit = False
        for m in pat.finditer(text):
            if any(m.start() < b2 and m.end() > a2 for a2, b2 in taken):
                continue
            taken.append((m.start(), m.end()))
            spans.append((m.start(), m.end(), e.get("stance", "")))
            hit = True
        if not hit:
            e["_unlocated"] = True
    out, pos = [], 0
    for a, b, st in sorted(spans):
        out.append(html.escape(text[pos:a]))
        bg = STANCE_BG.get(st, "#ffe9a8")
        out.append(f'<mark title="{html.escape(st)}" style="background:{bg};border-radius:3px;'
                   f'padding:0 2px">{html.escape(text[a:b])}</mark>')
        pos = b
    out.append(html.escape(text[pos:]))
    return "".join(out)


def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1")
    suffix = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
    prev = re.findall(r'data-rid="([^"]+)"', (HERE / "packets/demo_packet_judges.html").read_text())
    picked = sample20(load_corpus(), set(html.unescape(x) for x in prev))
    print(f"sampled {len(picked)} (excluded {len(prev)} packet responses)")
    for r in picked:
        print("  ", rid(r))

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda r: annotate(client, r["response"]), picked))

    rows, cards, tin = [], [], [0, 0]
    for i, (r, (ann, usage)) in enumerate(zip(picked, results)):
        ents = ann.get("entities", [])
        body = render(r["response"], ents)
        unloc = [e["surface"] for e in ents if e.get("_unlocated")]
        rows.append({"idx": i, "rid": rid(r), "condition": r["condition"],
                     "register": r["register"], "category": r["category"],
                     "model_key": r["model_key"], "response": r["response"],
                     "annotation": ann, "unlocated": unloc, "usage": usage})
        tin[0] += usage["tin"]; tin[1] += usage["tout"]
        chips = " ".join(
            f'<span style="background:{STANCE_BG.get(e.get("stance"), "#ffe9a8")};border-radius:8px;'
            f'padding:1px 7px;font-size:.8rem">{html.escape(e.get("surface") or "?")}</span>'
            for e in ents)
        flags = f'not_a_doctor={ann.get("not_a_doctor")} consult_professional={ann.get("consult_professional")}'
        un = f'<div style="color:#b45309;font-size:.8rem">unlocated: {html.escape(", ".join(unloc))}</div>' if unloc else ""
        cards.append(
            f'<div style="border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0">'
            f'<h3 style="margin:0 0 .3rem">#{i + 1} — {r["condition"]} / {r["register"]} / '
            f'{r["category"]} <small style="color:#666">({r["model_key"]})</small></h3>'
            f'<div style="margin:.3rem 0">{chips}</div>{un}'
            f'<div style="font-size:.8rem;color:#666;margin-bottom:.5rem">{flags}</div>'
            f'<div style="white-space:pre-wrap;line-height:1.5">{body}</div></div>')
        print(f"  #{i + 1} {rid(r)}: {len(ents)} entities" + (f" ({len(unloc)} unlocated)" if unloc else ""))

    (HERE / f"test20_luna{suffix}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    legend = "".join(f'<span style="background:{v};border-radius:3px;padding:1px 6px;margin-right:.5rem">{k}</span>'
                     for k, v in STANCE_BG.items()) + '<span style="background:#ffe9a8;border-radius:3px;padding:1px 6px">unmatched</span>'
    (HERE / f"test20_viewer{suffix}.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Luna test-20</title>'
        '<body style="max-width:60rem;margin:2rem auto;font-family:system-ui;padding:0 1rem">'
        f'<h2>Luna held-out test set — 20 unseen responses</h2><p>{legend}</p>'
        + "".join(cards) + "</body>")
    cost = tin[0] / 1e6 * 0.2 + tin[1] / 1e6 * 1.2
    print(f"done: {tin[0]} in / {tin[1]} out tokens, ~${cost:.2f}")
    print(f"wrote test20_luna{suffix}.jsonl + test20_viewer{suffix}.html")


if __name__ == "__main__":
    main()
