"""Shared definitions for the analysis chain.

Paths, the grid constants, the model-to-developer map, the lexicon mapper that links a
judge-extracted surface to an option on one of the four rating lists, and the one
definition every count in the audit rests on:

    RECOMMENDED = judge stance in {recommended, neutral, mixed}
    ADVISED_AGAINST = judge stance == advised_against

An intervention counts as recommended if the response names it without advising against
it. A neutral mention offered in answer to a request for options functions as a
recommendation; a mixed response recommended the option for at least one use. This is the
binary the human validation tested.

The scripts read the raw runs from a folder that holds runs/, as released with the full
corpus. Pass it with --runs or set MB_RUNS. Derived tables are written to data/.
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MB_DATA") or (ROOT / "data"))
INSTRUMENT = ROOT / "instrument"


def runs_root(arg=None) -> Path:
    """The folder that holds runs/: the --runs argument, else $MB_RUNS, else this repository.
    --dev and $MB_DEV are accepted as aliases."""
    return Path(arg or os.environ.get("MB_RUNS") or os.environ.get("MB_DEV") or ROOT)


FOOTINGS = ("clinical", "caregiver", "patient")
ASKS = ("open", "medication", "psychotherapy", "neuromodulation", "digital")
LISTS = ("pharmacological", "psychotherapy", "neuromodulation", "digital")
LIST_ASK = {"pharmacological": "medication", "psychotherapy": "psychotherapy",
            "neuromodulation": "neuromodulation", "digital": "digital"}
NULLS = {"pharmacological": "no medication", "psychotherapy": "no psychotherapy",
         "neuromodulation": "no somatic or neuromodulation treatment", "digital": "no digital intervention"}
CONTROLS = ("situational_stress_subclinical", "uncomplicated_bereavement")
CONDITIONS = tuple(k for k in json.loads((INSTRUMENT / "dataset/situations.json").read_text()) if not k.startswith("_"))
LANGUAGE = "en-US"   # language of the vignettes and option lists in this instrument version

RECOMMENDED_STANCES = {"recommended", "neutral", "mixed"}


def binary_stance(four_way: str) -> str:
    return "advised_against" if four_way == "advised_against" else "recommended"


_cfg = json.loads((ROOT / "config/models.json").read_text())["models"]
FAMILY = {k: v["family"] for k, v in _cfg.items()}
MODELS = tuple(_cfg)

# ---- lexicon: option names and the surfaces that map to them ----------------------
SURF2OPT = {}                     # (list, surface_lower) -> option_lower
OPTIONS = defaultdict(list)       # list -> [option_lower]
OPTION_DISPLAY = {}               # (list, option_lower) -> display name

_meds = json.loads((INSTRUMENT / "lexicon/psych_meds.json").read_text())
for g, info in _meds.items():
    OPTIONS["pharmacological"].append(g.lower())
    OPTION_DISPLAY[("pharmacological", g.lower())] = g
    for s in {g, *info.get("brands", []), *info.get("aliases", [])}:
        SURF2OPT[("pharmacological", s.lower())] = g.lower()
for _list, _fn in (("psychotherapy", "talk_therapies.json"), ("neuromodulation", "neuromodulation.json"),
                   ("digital", "digital.json")):
    for k, v in json.loads((INSTRUMENT / "lexicon" / _fn).read_text()).items():
        if not isinstance(v, dict) or k.startswith("_") or v.get("family") == "null_option":
            continue
        name = v["name"].lower()
        OPTIONS[_list].append(name)
        OPTION_DISPLAY[(_list, name)] = v["name"]
        for s in {v["name"], *v.get("aliases", []), *([v["abbrev"]] if v.get("abbrev") else [])}:
            SURF2OPT[(_list, s.lower())] = name

_DASHES = str.maketrans({"‐": "-", "‑": "-", "–": "-", "—": "-", "’": "'"})


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(_DASHES).strip().lower())


def map_surface(surface: str):
    """All (list, option) pairs a judge surface maps to: the whole surface, each part of a
    trailing parenthetical, and each slash- or plus-separated part, after dash and quote
    normalization. Usually zero or one pair; a compound surface can map to several."""
    cands = [norm(surface)]
    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", surface)
    if m:
        cands += [norm(m.group(1)), norm(m.group(2))]
    cands += [norm(p) for p in re.split(r"\s*/\s*|\s*\+\s*", surface) if p.strip()]
    hits = set()
    for c in cands:
        for lst in LISTS:
            if (lst, c) in SURF2OPT:
                hits.add((lst, SURF2OPT[(lst, c)]))
    return hits


def iter_judgments(runs_dir: Path):
    for line in (runs_dir / "runs/judge_v1/luna/judgments.jsonl").open():
        r = json.loads(line)
        if r.get("annotation") is not None and not r.get("error"):
            yield r


def iter_corpus(runs_dir: Path):
    """The frozen generation corpus the judge read: one row per response from the sidecar
    exports of both generation runs (rid, model, condition, footing, ask, sample, text)."""
    for arm in ("gen_v1", "legacy_v1"):
        for f in sorted((runs_dir / f"runs/{arm}/export/sidecar").glob("items-*.jsonl")):
            for line in f.open():
                r = json.loads(line)
                cond, footing, ask, sample = r["item_key"].split("|")
                mk = r["raw"]["model_key"]
                yield {"rid": f"{mk}|{r['item_key']}", "model": mk, "condition": cond, "footing": footing,
                       "ask": ask, "sample": sample, "text": r["output"]}


def read_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
