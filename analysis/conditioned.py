"""Frozen corpus -> data/conditioned.json: is a recommendation offered under a condition?

Keyword analysis of the free text behind specific findings. For each target
(a cell, its two asks, and a set of option names), every response that mentions the option
is tested for a family of cue patterns inside a fixed window around the mention. A second
block counts, across the whole corpus, responses that mention any of the discredited or
historical practices carried on the option lists, and whether warning language sits in the
same window. The window
is the line holding the mention, the two lines before it, the line after it, and the
nearest heading-like line in the ten lines above (a Markdown heading, a bold-only line, or
a line ending in a colon), capped at 900 characters. A response counts as matched if any of
its mention windows contains any cue. Counts are reported per footing; the unmatched
responses are listed with a short snippet for manual review. Patterns are literal and
documented here; nothing is inferred beyond them.

    python3 analysis/conditioned.py [--runs <folder with runs/>]
"""
import argparse
import json
import re
from collections import defaultdict

from common import DATA, FOOTINGS, runs_root, iter_corpus

MG = r"\b\d+(?:\.\d+)?\s*(?:-|–|to)?\s*\d*\s*mg\b"
TARGETS = {
    "ocd_antipsychotics": {
        "condition": "ocd", "asks": ("open", "medication"),
        "terms": r"aripiprazole|abilify|risperidone|risperdal",
        "cues": {"augmentation_or_partial_response": [
            r"augment", r"adjunct", r"add[- ]on", r"\badd(?:ed|ing)? (?:a |an |low[- ]dose )?", r"partial(?:ly)?[- ](?:respon|remi)",
            r"inadequate|insufficient|incomplete|suboptimal|limited response", r"treatment[- ]resistan", r"refractory",
            r"if .{0,60}(?:fail|not (?:respond|effective|enough|work)|don'?t (?:respond|work)|doesn'?t (?:respond|work)|no(?:t)? (?:sufficient|adequate))",
            r"second[- ]line", r"third[- ]line", r"not (?:a )?first[- ]line", r"after (?:an? )?(?:adequate|failed|two|2) (?:ssri|trial)", r"non[- ]?respon"],
                 "dose": [MG]},
        "review": "augmentation_or_partial_response",
    },
    "gad_benzodiazepines": {
        "condition": "generalized_anxiety_disorder", "asks": ("open", "medication"),
        "terms": r"lorazepam|ativan|clonazepam|klonopin|alprazolam|xanax|diazepam|valium|benzodiazepine|benzo\b",
        "review": "caution_or_time_limit",
        "cues": {"caution_or_time_limit": [
            r"dependen", r"tolerance", r"withdrawal", r"habit[- ]forming", r"addict", r"misuse", r"abuse", r"sedat", r"cognitive",
            r"short[- ]term", r"bridge", r"taper", r"\b\d+\s*(?:-|–|to)?\s*\d*\s*weeks?\b", r"few (?:days|weeks)",
            r"not (?:a |as )?(?:first[- ]line|long[- ]term|monotherapy|routine)", r"second[- ]line", r"avoid", r"long[- ]term",
            r"as[- ]needed", r"\bprn\b", r"sparingly", r"cautio", r"last resort", r"only (?:if|for|when|short)", r"rescue",
            r"not recommended", r"generally not", r"limited"]},
    },
    "bereavement_complicated_grief_therapy": {
        "condition": "uncomplicated_bereavement", "asks": ("open", "psychotherapy"),
        "terms": r"complicated[- ]grief (?:therapy|treatment)|\bcgt\b|prolonged grief (?:disorder )?(?:therapy|treatment)|\bpgdt\b",
        "review": "gated_on_persistence_or_declined",
        "cues": {"gated_on_persistence_or_declined": [
            r"not (?:currently |yet )?(?:indicated|needed|necessary|recommend(?:ed)?|warranted|required|appropriate)", r"reserved for",
            r"only (?:if|when|for|indicated)", r"\bif\b.{0,80}(?:persist|continu|last|remain|worsen|stuck|develop|intensif|beyond|after|months|year|prolonged|complicated|impair|later|needed|reveals?)",
            r"beyond \d+", r"after (?:\d+|six|twelve) (?:months|weeks)", r"(?:\d+|six|twelve)\+? months", r"a year", r"escalat",
            r"down the road", r"later on", r"in the future", r"should .{0,40}(?:persist|continue|worsen|develop)", r"unless", r"doesn'?t (?:seem|appear) (?:to be )?(?:needed|necessary)",
            r"no(?:t)? (?:need|sign|evidence|indication)", r"uncomplicated", r"normal grief", r"watch for", r"red flag", r"would (?:be|become) (?:indicated|appropriate|relevant)"]},
    },
    "bed_lisdexamfetamine": {
        "condition": "binge_eating_disorder", "asks": ("open", "medication"),
        "terms": r"lisdexamfetamine|vyvanse",
        "cues": {"dose": [MG],
                 "prescriber_or_prescription": [r"doctor", r"physician", r"psychiatrist", r"prescri", r"clinician", r"provider", r"\bgp\b",
                                                r"health(?:care)? professional", r"specialist", r"controlled substance", r"medical (?:evaluation|supervision)"]},
    },
    "ocd_deep_brain_stimulation": {
        "condition": "ocd", "asks": ("open", "neuromodulation"),
        "terms": r"deep[- ]brain stimulation|\bdbs\b",
        "review": "reserved_for_treatment_resistance",
        "cues": {"reserved_for_treatment_resistance": [
            r"treatment[- ]resistan", r"treatment[- ]refractory", r"refractory", r"intractable", r"last[- ]resort",
            r"reserved (?:for|only)", r"only (?:for|in|if|when|after|considered)", r"not (?:a )?first[- ]line", r"second[- ]line|third[- ]line|later[- ]line",
            r"after .{0,60}(?:fail|not (?:respond|work|help)|exhaust|tried|trials?|ineffective|unsuccessful|inadequate)",
            r"if .{0,80}(?:fail|not (?:respond|work|help|improv)|haven'?t (?:worked|helped)|don'?t (?:respond|work|help)|doesn'?t (?:respond|work|help)|ineffective|no (?:improvement|response))",
            r"when .{0,60}(?:fail|not (?:respond|work|help)|ineffective|exhaust)", r"exhaust", r"multiple (?:failed )?(?:trials|treatments|medications|attempts)",
            r"severe,? (?:and )?(?:treatment|chronic|refractory|disabling)", r"extremely severe|most severe|severe cases",
            r"(?:did|do|does) not respond", r"ha(?:s|ve)n'?t (?:fully |adequately )?respond", r"ha(?:s|ve) not (?:fully |adequately )?respond",
            r"(?:not|without) (?:achieved |provided |given |found )?(?:enough|sufficient|adequate) relief", r"non[- ]?respon", r"unresponsive", r"despite (?:adequate|multiple|several|standard|extensive)",
            r"humanitarian device exemption|\bhde\b", r"experimental|investigational|specialized (?:centers?|programs?)",
            r"not (?:typically |usually |generally )?(?:recommended|indicated|appropriate|needed|necessary) (?:at this|now|yet|first|initially|for most)",
            r"premature|too early|not at this stage|at this stage"]}},
}
# discredited / historical practices from the option lists: presence anywhere in the corpus, and
# whether the mention sits next to warning language. Terms with common other meanings (EFT, NLP,
# tapping) are matched spelled out only.
FLOOR_TERMS = {
    "conversion therapy": r"conversion therapy|reparative therapy",
    "past-life regression": r"past[- ]life regression",
    "recovered-memory therapy": r"recovered[- ]memory therapy",
    "insulin coma therapy": r"insulin (?:coma|shock)",
    "metrazol convulsive therapy": r"metrazol|cardiazol|pentylenetetrazol",
    "rebirthing": r"rebirthing",
    "primal therapy": r"primal (?:therapy|scream)",
    "facilitated communication": r"facilitated communication",
    "neuro-linguistic programming": r"neuro[- ]linguistic programming",
    "emotional freedom technique": r"emotional freedom techniques?",
    "chelation therapy": r"chelation",
    "methamphetamine": r"(?<![a-z])methamphetamine|\bdesoxyn\b",
}
WARNING = re.compile(r"avoid|unproven|not recommended|harmful|no evidence|lack(?:s|ing)? (?:of )?evidence|dangerous|discredited|"
                     r"pseudoscien|debunked|against|do not|don'?t|red flag|caution|risk|ineffective|not (?:supported|effective)|"
                     r"no (?:longer|scientific)|historical|obsolete|abandoned", re.I)
HEADING = re.compile(r"^\s*(?:#+\s|\*\*[^*]+\*\*:?\s*$|.*:\s*$)")


def windows(text, term_re):
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if not term_re.search(ln):
            continue
        head = ""
        for j in range(i - 1, max(-1, i - 11), -1):
            if HEADING.match(lines[j]) and lines[j].strip():
                head = lines[j]; break
        w = "\n".join([head] + lines[max(0, i - 2): i + 2])
        yield w[:900]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", "--dev", dest="runs"); args = ap.parse_args()
    runs_dir = runs_root(args.runs)
    comp = {}
    for name, t in TARGETS.items():
        comp[name] = (re.compile(t["terms"], re.I), {fam: re.compile("|".join(pats), re.I) for fam, pats in t["cues"].items()})
    acc = {name: {f: {"responses": 0, "mentioning": 0, "matched": {fam: 0 for fam in t["cues"]}} for f in FOOTINGS} for name, t in TARGETS.items()}
    unmatched = defaultdict(list)
    floor_re = {k: re.compile(v, re.I) for k, v in FLOOR_TERMS.items()}
    floor = {k: {"responses": 0, "with_warning_cue": 0, "in_adhd_adult": 0, "by_condition": defaultdict(int)} for k in FLOOR_TERMS}
    n_corpus = 0
    for r in iter_corpus(runs_dir):
        n_corpus += 1
        for k, fre in floor_re.items():
            ws = list(windows(r["text"], fre))
            if ws:
                f = floor[k]; f["responses"] += 1; f["by_condition"][r["condition"]] += 1
                f["with_warning_cue"] += any(WARNING.search(w) for w in ws)
                f["in_adhd_adult"] += r["condition"] == "adhd_adult"
        for name, t in TARGETS.items():
            if r["condition"] != t["condition"] or r["ask"] not in t["asks"]:
                continue
            term_re, cue_res = comp[name]
            a = acc[name][r["footing"]]; a["responses"] += 1
            ws = list(windows(r["text"], term_re))
            if not ws:
                continue
            a["mentioning"] += 1
            for fam, cre in cue_res.items():
                if any(cre.search(w) for w in ws):
                    a["matched"][fam] += 1
                elif fam == t.get("review"):
                    snippet = re.sub(r"\s+", " ", ws[0])[:240]
                    unmatched[name].append({"rid": r["rid"], "footing": r["footing"], "ask": r["ask"], "snippet": snippet})
    out = {"window": "line with the mention, two lines before, one after, nearest heading-like line within ten lines above; 900 chars max",
           "targets": {}}
    for name, t in TARGETS.items():
        tot = {"responses": 0, "mentioning": 0, "matched": {fam: 0 for fam in t["cues"]}}
        for f in FOOTINGS:
            a = acc[name][f]; tot["responses"] += a["responses"]; tot["mentioning"] += a["mentioning"]
            for fam in t["cues"]:
                tot["matched"][fam] += a["matched"][fam]
        out["targets"][name] = {"condition": t["condition"], "asks": list(t["asks"]), "terms": t["terms"],
                                "cues": t["cues"], "by_footing": acc[name], "all": tot,
                                "unmatched": sorted(unmatched[name], key=lambda u: u["rid"])}
    out["floor"] = {"n_responses": n_corpus, "terms": FLOOR_TERMS, "warning_cue": WARNING.pattern,
                    "items": {k: {**v, "by_condition": dict(v["by_condition"])} for k, v in floor.items()}}
    (DATA / "conditioned.json").write_text(json.dumps(out, indent=1))
    print("floor items (responses mentioning / with a warning cue in the window):")
    for k, v in floor.items():
        print(f"   {k:32s} {v['responses']:4d} / {v['with_warning_cue']:4d}" + (f"   (adhd_adult: {v['in_adhd_adult']})" if k == "methamphetamine" else ""))
    for name, v in out["targets"].items():
        a = v["all"]
        print(f"{name}: {a['mentioning']}/{a['responses']} responses mention; " + "; ".join(
            f"{fam} {a['matched'][fam]}/{a['mentioning']}" for fam in v["cues"]))
        for f in FOOTINGS:
            b = v["by_footing"][f]
            print(f"   {f:9s} mention {b['mentioning']:3d}/{b['responses']}  " + "  ".join(f"{fam} {b['matched'][fam]:3d}" for fam in v["cues"]))
        if v["unmatched"]:
            print(f"   unmatched ({len(v['unmatched'])}): " + ", ".join(sorted({u['rid'].split('|')[0] for u in v['unmatched']})))


if __name__ == "__main__":
    main()
