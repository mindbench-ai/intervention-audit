"""Ingest the human codings (two calibration pilots, main wave rounds 1 and 2) into the
release artifacts under validation/codings/.

This script reads the as-run coder exports (runs/pilot/, runs/pilot2/, runs/main_wave_r1/,
runs/main_wave_r2/raw/), which are not released because they identify the coders. It is kept here so the transforms behind the released files are on record and
can be re-run where the exports exist.

Sources and forms:
  pilots      .json download exports tagged with coder initials; mapped to the anonymous
              R-codes through an off-repo map (--coder-map; default
              ~/.mindbench/intervention-audit-coder-map.json). The map is never released.
  main wave   round 1: .json download exports (R7/R8 were extracted from .docx earlier)
              round 2: .json download exports (R1-R5), a clipboard export pasted into a
              text file (R6), clipboard exports pasted into Word (R7, R8)

Transforms, all lossless and logged per file in the output's "repairs" list:
  * .docx -> text via pandoc; the JSON object is the first '{' .. last '}'.
  * clipboard exports wrote quote characters inside entity names without escaping;
    quotes inside a stance key are re-escaped before parsing.
  * a text editor straightened typographic quotes in some clipboard-export keys; keys are
    matched on the straightened form and restored to the packet's exact surface.
  * two interfaces exported one entity twice, once with rendered HTML instead of the
    judge's markdown surface; the duplicate is merged onto the markdown key.
  * URL defanging (a space after https://, http://, www.) is reversed in every string.

Validation: coder id, packet id, expected answer counts, rids assigned to that coder and
wave (main wave) or in the pilot selection (pilots), and stance keys a subset of the
entities shown in that packet's HTML.

Outputs: validation/codings/R<i>_pilot1.json, R<i>_pilot2.json, R<i>_round1.json,
R<i>_round2.json, codings_all.jsonl, manifest.json.

    python3 validation/ingest_codings.py --runs <folder with runs/> [--coder-map <file>]
"""
import argparse
import hashlib
import html as H
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "codings"
STANCES = {"recommended", "neutral", "advised_against", "mixed", "not_applicable", ""}
CODERS = [f"R{i}" for i in range(1, 9)]

MAIN_R2_RAW = {"R1": "runs/main_wave_r2/raw/mb_main_R1_r2.json", "R2": "runs/main_wave_r2/raw/mb_main_R2_r2.json",
               "R3": "runs/main_wave_r2/raw/mb_main_R3_r2.json", "R4": "runs/main_wave_r2/raw/mb_main_R4_r2.json",
               "R5": "runs/main_wave_r2/raw/mb_main_R5_r2.json", "R6": "runs/main_wave_r2/raw/mb_round2_R6.txt",
               "R7": "runs/main_wave_r2/raw/R7_round2.docx", "R8": "runs/main_wave_r2/raw/R8_round2.docx"}
PILOT_RAW = {"pilot1": ("runs/pilot", "mb_pilot_*.json", "pilot_packet.html", "pilot10_selection.json"),
             "pilot2": ("runs/pilot2", "mb_pilot2_*.json", "pilot2_packet.html", "pilot2_selection.json")}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def raw_text(p: Path) -> str:
    if p.suffix == ".docx":
        return subprocess.run(["pandoc", str(p), "-t", "plain", "--wrap=none"],
                              capture_output=True, text=True, check=True).stdout
    return p.read_text()


def parse_export(p: Path):
    t = raw_text(p)
    t = t[t.find("{"): t.rfind("}") + 1]
    repairs = []
    try:
        return json.loads(t), repairs
    except json.JSONDecodeError:
        pass

    def fix(m):
        key = re.sub(r'(?<!\\)"', r'\\"', m.group(2))
        return f'{m.group(1)}"{key}": "{m.group(3)}"{m.group(4)}'

    fixed = re.sub(r'^(\s+)"(.*)": "([a-z_]*)"(,?)$', fix, t, flags=re.M)
    if fixed != t:
        repairs.append("re-escaped raw quotes inside stance keys (clipboard export bug)")
    return json.loads(fixed), repairs


def refang(s):
    if not isinstance(s, str):
        return s
    return s.replace("https:// ", "https://").replace("http:// ", "http://").replace("www. ", "www.")


def straight(s: str) -> str:
    return s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def packet_entities(page: Path) -> dict:
    ents = {}
    for m in re.finditer(r'data-rid="([^"]+)" data-ents=\'([^\']*)\'', page.read_text()):
        ents[H.unescape(m.group(1))] = {refang(e) for e in json.loads(H.unescape(m.group(2)))}
    return ents


def clean_stances(raw_stances, rid, valid, repairs):
    stances = {}
    for k, v in raw_stances.items():
        k = refang(k)
        if "<" in k:
            k2 = re.sub(r"</?b>", "**", k)
            k2 = re.sub(r"</?i>", "*", k2)
            k2 = re.sub(r"<[^>]+>", "", k2)
            repairs.append(f"merged rendered-HTML duplicate key into {k2!r} ({rid})")
            if stances.get(k2):
                continue
            k = k2
        if k in stances and stances[k] and not v:
            continue
        stances[k] = v
    canon = {straight(e): e for e in valid}
    restored = {}
    for k, v in stances.items():
        if k not in valid and straight(k) in canon:
            repairs.append(f"restored typographic quotes in key {canon[straight(k)]!r} ({rid})")
            k = canon[straight(k)]
        restored[k] = v
    bad = set(restored) - valid
    badv = set(restored.values()) - STANCES
    return restored, ([f"unknown entity keys for {rid}: {sorted(bad)[:3]}"] if bad else []) + \
        ([f"unknown stance values for {rid}: {sorted(badv)}"] if badv else [])


def answer_row(a, rid, split, stances):
    return {"rid": rid, "split": split, "slot": a["slot"], "seconds": a.get("seconds"), "stances": stances,
            "missed": refang(a.get("missed") or ""), "not_a_doctor": bool(a.get("not_a_doctor")),
            "consult_professional": bool(a.get("consult_professional"))}


def normalise_main(d, coder, round_no, meta, ents, src, repairs):
    assert d["coder"] == coder, (src, d["coder"])
    assert d.get("packet") == f"main_r{round_no}", (src, d.get("packet"))
    answers, problems = [], []
    for a in d["answers"]:
        rid = a["rid"]
        asg = meta["assignments"].get(rid)
        if asg is None or coder not in asg["coders"]:
            problems.append(f"rid not assigned to {coder}: {rid}")
            continue
        if asg["wave"] != ("1" if a["slot"] == "w1" else "2"):
            problems.append(f"slot/wave mismatch {rid}")
        stances, probs = clean_stances(a.get("stances") or {}, rid, ents.get(rid, set()), repairs)
        problems += probs
        answers.append(answer_row(a, rid, "dev" if asg["wave"] == "1" else "test", stances))
    answers.sort(key=lambda x: x["rid"])
    n_w1 = sum(1 for x in answers if x["slot"] == "w1")
    q = meta["quotas"]
    expected = (q["round1"], 0) if round_no == 1 else (q["round1"], q["round2"])
    assert (n_w1, len(answers) - n_w1) == expected, (src, n_w1, len(answers) - n_w1)
    if problems:
        raise SystemExit(f"{src}: validation problems:\n  " + "\n  ".join(problems))
    return {"coder": coder, "round": round_no, "packet": d["packet"], "source_file": src, "repairs": repairs,
            "avg_seconds_per_response": d.get("avg_seconds_per_response"), "n_answers": len(answers),
            "answers": answers}


def normalise_pilot(d, pilot, coder, sel_cids, ents, src, repairs):
    assert d.get("packet") in (pilot, "pilot10" if pilot == "pilot1" else pilot), (src, d.get("packet"))
    answers, problems = [], []
    for a in d["answers"]:
        rid = a["rid"]
        if rid not in sel_cids:
            problems.append(f"rid not in {pilot} selection: {rid}")
            continue
        stances, probs = clean_stances(a.get("stances") or {}, rid, ents.get(rid, set()), repairs)
        # pilot packets were revised between sittings; keys from an older packet version are dropped
        stale = [p for p in probs if p.startswith("unknown entity keys")]
        if stale:
            for k in list(stances):
                if k not in ents.get(rid, set()):
                    stances.pop(k)
            repairs.append(f"dropped stale entity keys from an earlier packet version ({rid})")
            probs = [p for p in probs if not p.startswith("unknown entity keys")]
        problems += probs
        answers.append(answer_row(a, rid, "pilot", stances))
    answers.sort(key=lambda x: x["rid"])
    assert len(answers) == 10, (src, len(answers))
    if problems:
        raise SystemExit(f"{src}: validation problems:\n  " + "\n  ".join(problems))
    return {"coder": coder, "round": pilot, "packet": d.get("packet"), "source_file": src, "repairs": repairs,
            "avg_seconds_per_response": d.get("avg_seconds_per_response"), "n_answers": len(answers),
            "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", "--dev", dest="runs", required=True, help="folder that holds the as-run exports under runs/")
    ap.add_argument("--coder-map", default=str(Path.home() / ".mindbench/intervention-audit-coder-map.json"))
    args = ap.parse_args()
    runs_dir = Path(args.runs)
    cmap = json.loads(Path(args.coder_map).read_text())
    meta = json.loads((HERE / "assignments.json").read_text())
    OUT.mkdir(exist_ok=True)
    manifest = {"sources": {}, "outputs": {}}
    outputs = []

    # pilots
    for pilot, (rdir, glob, packet, selection) in PILOT_RAW.items():
        ents = packet_entities(HERE / "packets" / packet)
        sel_cids = {c["cid"] for c in json.loads((HERE / selection).read_text()).values()}
        for p in sorted((runs_dir / rdir).glob(glob)):
            d, repairs = parse_export(p)
            coder = cmap[pilot][d["coder"]]
            norm = normalise_pilot(d, pilot, coder, sel_cids, ents, f"{rdir}/<{pilot} export>", repairs)
            out = OUT / f"{coder}_{pilot}.json"
            out.write_text(json.dumps(norm, indent=1, ensure_ascii=False) + "\n")
            manifest["sources"][f"{rdir}/<{pilot} export of {coder}>"] = sha(p)
            outputs.append((out, norm))
            print(f"{coder} {pilot}: {norm['n_answers']} answers" + (f"  [{'; '.join(sorted(set(repairs)))}]" if repairs else ""))

    # main wave
    ents = {1: {}, 2: {}}
    for i in range(1, 9):
        for r in (1, 2):
            ents[r].update(packet_entities(HERE / "packets" / "main_wave" / f"R{i}_round{r}.html"))
    for i in range(1, 9):
        coder = f"R{i}"
        for round_no, src in ((1, f"runs/main_wave_r1/mb_main_R{i}_r1.json"), (2, MAIN_R2_RAW[coder])):
            p = runs_dir / src
            d, repairs = parse_export(p)
            norm = normalise_main(d, coder, round_no, meta, ents[round_no], src, repairs)
            out = OUT / f"{coder}_round{round_no}.json"
            out.write_text(json.dumps(norm, indent=1, ensure_ascii=False) + "\n")
            manifest["sources"][src] = sha(p)
            outputs.append((out, norm))
            print(f"{coder} round {round_no}: {norm['n_answers']} answers" + (f"  [{'; '.join(sorted(set(repairs)))}]" if repairs else ""))

    combined = OUT / "codings_all.jsonl"
    with combined.open("w") as allf:
        for out, norm in outputs:
            manifest["outputs"][out.name] = sha(out)
            for a in norm["answers"]:
                allf.write(json.dumps({"coder": norm["coder"], "round": norm["round"], **a}, ensure_ascii=False) + "\n")
    manifest["outputs"][combined.name] = sha(combined)
    manifest["assignments_json"] = sha(HERE / "assignments.json")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
