"""Structured-rating instrument: option lists, prompt composition, provider call, parser.

Shared by run/rate_qualify.py (qualification probe), run/rate_run.py (live lanes) and
run/batch_rate_run.py (batch lanes). Nothing here runs on its own.

A rating prompt is one user message with no system prompt:

    == PERSON / SITUATION ==
    <presentation in the cell's footing>

    == OPTIONS TO RATE ==
    <every option on the class list plus its null option, comma-separated, in a
     seeded random order that differs per cell>

    <instruction composed from instrument/lexicon/anchors.json: the framing phrase, the
     temporal clause, the seven scale anchors, the null note, the output format>

The model answers with a JSON object mapping option names to integer scores. The parser
is truncation-tolerant: it extracts every "name": score pair it can find, so a response
cut off mid-object still yields its completed ratings.

anchors.json keeps its framing/temporal dictionaries with a default key so that new
framings or voices can be added without changing this code; the audited run used the
"neutral" framing and the "initial" temporal clause only.
"""

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT = ROOT / "instrument"

SIT = {k: v for k, v in json.loads((INSTRUMENT / "dataset/situations.json").read_text()).items()
       if not k.startswith("_")}
ANCH = json.loads((INSTRUMENT / "lexicon/anchors.json").read_text())

CLASSES = ["pharmacological", "psychotherapy", "neuromodulation", "digital"]
CLASS_FILE = {"pharmacological": "psych_meds.json", "psychotherapy": "talk_therapies.json",
              "neuromodulation": "neuromodulation.json", "digital": "digital.json"}


def class_options(cls):
    """Display names of the options on one class list, null option excluded."""
    d = json.loads((INSTRUMENT / "lexicon" / CLASS_FILE[cls]).read_text())
    if cls == "pharmacological":
        return list(d.keys())  # medications are keyed by generic name
    return [v["name"] for k, v in d.items()
            if isinstance(v, dict) and not k.startswith("_") and v.get("family") != "null_option"]


def scope_list(scope):
    """Options presented for one class: the list plus its explicit null option."""
    return class_options(scope) + [ANCH["per_class_null"][scope]]


def build_instruction(temporal="initial"):
    return ANCH["instruction_template"].format(
        framing_word=ANCH["framing_words"][ANCH["default_framing"]],
        temporal_clause=ANCH["temporal"][temporal],
        scale="\n".join(ANCH["scale"]), null_note=ANCH["null_note"])


def build_message(cond, footing, scope, temporal, seed):
    """The full prompt for one cell and the number of options in it. `seed` fixes the
    option order; the runners pass "<model>|<condition>|<footing>|<scope>|<sample>"
    (the probe omits the footing, which is always clinical there)."""
    names = scope_list(scope)[:]
    random.Random(seed).shuffle(names)
    return (f"== PERSON / SITUATION ==\n{SIT[cond][footing]}\n\n== OPTIONS TO RATE ==\n"
            f"{', '.join(names)}\n\n{build_instruction(temporal)}"), len(names)


def name_lower_map(scope):
    return {n.lower(): n for n in scope_list(scope)}


def parse_ratings(text, lmap):
    """Map every '"name": score' pair in the response to its canonical option name.
    Works on truncated JSON; names not on the list are ignored."""
    if not text:
        return {}
    out = {}
    for m in re.finditer(r'"([^"]+)"\s*:\s*(-?[0-3])(?![0-9])', text):
        canon = lmap.get(m.group(1).strip().lower())
        if canon is not None:
            out[canon] = int(m.group(2))
    return out


def _g(obj, *path):
    for p in path:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


async def call(provider, client, mid, msg, maxtok):
    """One rating call. Returns (text, telemetry). For OpenAI-compatible providers the
    request parameters are adjusted cumulatively on rejection: the token parameter name
    (max_completion_tokens for newer OpenAI models), the cap (some providers refuse
    32,768 and accept 8,192 or 4,096), and thinking (DashScope-served Qwen3 needs it
    disabled). A provider can need more than one of these, so they accumulate rather than
    being tried one at a time."""
    if provider == "anthropic":
        r = await client.messages.create(model=mid, max_tokens=maxtok,
                                         messages=[{"role": "user", "content": msg}])
        text = "".join(b.text for b in r.content if b.type == "text")
        u = r.usage
        return text, {"tokens_in": u.input_tokens, "tokens_out": u.output_tokens,
                      "finish_reason": r.stop_reason, "served_model": r.model}
    cap, param, extra = maxtok, "max_tokens", None
    for _ in range(5):
        try:
            kw = {param: cap}
            if extra:
                kw["extra_body"] = extra
            r = await client.chat.completions.create(model=mid, messages=[{"role": "user", "content": msg}], **kw)
            break
        except Exception as e:
            es = str(e)
            if "max_completion_tokens" in es and param == "max_tokens":
                param = "max_completion_tokens"
            elif "max_tokens is too large" in es and cap > 4096:
                cap = 4096
            elif "Range of max_tokens" in es and cap > 8192:
                cap = 8192
            elif "enable_thinking" in es and not extra:
                extra = {"enable_thinking": False}
            else:
                raise
    ch = r.choices[0]
    u = r.usage
    return (ch.message.content or ""), {
        "reasoning_content": _g(ch, "message", "reasoning_content"),
        "tokens_in": getattr(u, "prompt_tokens", None), "tokens_out": getattr(u, "completion_tokens", None),
        "tokens_reasoning": _g(r, "usage", "completion_tokens_details", "reasoning_tokens"),
        "finish_reason": ch.finish_reason, "served_model": r.model}
