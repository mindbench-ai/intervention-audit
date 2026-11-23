"""Verify the frozen instrument pack against its expert-review source spreadsheets.

The runtime files the runners read (dataset/*.json, lexicon/*.json) were generated from
the clinician-reviewed CSVs in expert_review/ and frozen as pack v1.0.0 on 2026-08-21
(VERSION.json holds the sha256 prefix of every file). This script rebuilds each runtime
file from its CSV in memory and checks that the result matches the frozen file, and that
every frozen file still matches VERSION.json. It writes nothing.

The expert_review sheets in this repository are copies of the frozen sheets with reviewer
names replaced by letters and internal working notes removed; their content columns are
unchanged. VERSION.json keeps each sheet's frozen hash (sha256_16) and adds the hash of the
copy here (public_sha256_16); a sheet passes if it matches either, so the same check runs
on the frozen sheets and on these copies. The runtime files rebuild exactly from both.

    python3 instrument/check_pack.py        (run from anywhere; exits non-zero on mismatch)

Mapping (CSV -> runtime file):
  01d_patient_scenarios_v4.csv -> dataset/situations.json
  02d_asks_v4.csv              -> dataset/asks.json
  05c_drugs_generic_v2.csv     -> lexicon/psych_meds.json
  07b_digital_apps_v2.csv      -> lexicon/digital.json
  08b_brain_stimulation_v2.csv -> lexicon/neuromodulation.json
  09b_talk_therapies_v2.csv    -> lexicon/talk_therapies.json

asks.json is compared on its ask entries only: the frozen file also carries a "_deferred"
block holding two asks that were designed and dropped before any run. They were never
sent to a model and are not in the CSV; they remain in the frozen file because its hash
is part of the pack manifest.
"""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ER = HERE / "expert_review"
VERSION = "1.0.0"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def split(s):
    return [a.strip() for a in (s or "").split(";") if a.strip()]


def read(name):
    with open(ER / name, newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def build_situations():
    rows = read("01d_patient_scenarios_v4.csv")
    out = {"_doc": f"v{VERSION} scenarios, clinician-signed-off (from 01d_patient_scenarios_v4.csv). "
                   "Registers: clinical / caregiver / patient. Content matched across registers; "
                   "style profiles in expert_review/style-profiles-v3.md.",
           "_controls": []}
    for r in rows:
        cond = r["condition"].strip()
        out.setdefault(cond, {})[r["arm"].strip()] = r["scenario_text"].strip()
        if (r.get("no_intervention_control?") or "").strip() and cond not in out["_controls"]:
            out["_controls"].append(cond)
    return out


def build_asks():
    rows = read("02d_asks_v4.csv")
    out = {}
    for r in rows:
        out.setdefault(r["ask_category"].strip(), {})[r["register"].strip()] = r["ask_text"].strip()
    return out


def build_meds():
    out = {}
    for r in read("05c_drugs_generic_v2.csv"):
        out[r["generic"].strip()] = {"class": r["drug_class"].strip(),
                                     "brands": split(r["brands_listed"]),
                                     "aliases": split(r["aliases"])}
    return out


def build_digital():
    out = {}
    for r in read("07b_digital_apps_v2.csv"):
        name = r["name"].strip()
        tag = (r.get("internal_regulatory_tag (HIDDEN from models)") or "").strip()
        out[slug(name)] = {"name": name,
                           "family": "null_option" if name.lower().startswith("no digital") else "digital",
                           "regulatory": tag or None,
                           "availability": (r.get("availability (internal)") or "").strip() or None,
                           "aliases": split(r["aliases"])}
    return out


def build_neuromod():
    out = {}
    for r in read("08b_brain_stimulation_v2.csv"):
        name = r["name"].strip()
        out[slug(name)] = {"name": name, "abbrev": (r.get("abbrev") or "").strip() or None,
                           "family": r["family (internal)"].strip() or "null_option",
                           "invasiveness_tier": (r.get("invasiveness_tier (internal)") or "").strip() or None,
                           "aliases": split(r["aliases"]),
                           "floor": (r.get("intended_FLOOR_item?") or "").strip().lower() == "yes"}
    return out


def build_therapies():
    out = {}
    for r in read("09b_talk_therapies_v2.csv"):
        name = r["name"].strip()
        out[slug(name)] = {"name": name, "abbrev": (r.get("abbrev") or "").strip() or None,
                           "family": r["family (internal)"].strip() or "null_option",
                           "aliases": split(r["aliases"]),
                           "floor": (r.get("fringe/floor_family?") or "").strip().lower() == "yes"}
    return out


def main():
    manifest = json.loads((HERE / "VERSION.json").read_text())
    failures = []

    # 1. every file in the manifest still has its frozen hash (a review sheet may instead
    #    match the hash of its published copy)
    for rel, info in manifest["files"].items():
        h = sha(HERE / rel)
        ok = h in (info["sha256_16"], info.get("public_sha256_16"))
        copy = f"  (public copy of {info['sha256_16']})" if h == info.get("public_sha256_16") else ""
        print(f"  {'ok ' if ok else 'BAD'} manifest  {rel}  {h}{copy}")
        if not ok:
            failures.append(f"hash mismatch: {rel}")

    # 2. every runtime file matches a fresh build from its CSV
    targets = {"dataset/situations.json": build_situations, "dataset/asks.json": build_asks,
               "lexicon/psych_meds.json": build_meds, "lexicon/digital.json": build_digital,
               "lexicon/neuromodulation.json": build_neuromod, "lexicon/talk_therapies.json": build_therapies}
    for rel, fn in targets.items():
        frozen = json.loads((HERE / rel).read_text())
        built = fn()
        if rel == "dataset/asks.json":
            frozen = {k: v for k, v in frozen.items() if not k.startswith("_")}
        ok = frozen == built
        print(f"  {'ok ' if ok else 'BAD'} rebuild   {rel}  ({sum(1 for k in built if not k.startswith('_'))} entries)")
        if not ok:
            failures.append(f"rebuild differs: {rel}")

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print(f"\npack v{manifest['version']} verified: {len(manifest['files'])} files match the manifest, "
          f"{len(targets)} runtime files match their CSVs")


if __name__ == "__main__":
    main()
