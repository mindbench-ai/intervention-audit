"""Frozen corpus -> data/authorities.json: which authorities and justification frames the
responses invoke.

A deterministic string search over all 35,055 responses. The list was fixed before counting
and spans national medicines regulators from six jurisdictions, device marking, guideline
and professional bodies for psychiatry, psychology, and sleep medicine, public health and
veterans' agencies, app review bodies, evidence sources, nosologies, privacy law, and the
bare justification phrases models use ("evidence-based", "gold standard", "first-line",
"approved", "cleared", "off-label"). For each entry: the share of responses containing a
match, overall, by footing, by ask, and by developer. A match is a mention; it does not
say the reference was used correctly. Whole-word, case-insensitive; a few acronyms carry
context guards to avoid common-word collisions (NICE, WHO, VA, APA is reported as the bare
acronym and as each association separately).

    python3 analysis/authorities.py [--runs <folder with runs/>]
"""
import argparse
import json
import re
from collections import Counter, defaultdict

from common import ASKS, DATA, FAMILY, FOOTINGS, runs_root, iter_corpus

REFERENCES = {
    # medicines regulators and device marking
    "FDA": r"\bFDA\b|food and drug administration",
    "EMA": r"\bEMA\b|european medicines agency",
    "MHRA": r"\bMHRA\b",
    "Health Canada": r"health canada",
    "TGA": r"\bTGA\b|therapeutic goods administration",
    "PMDA": r"\bPMDA\b",
    "CE mark": r"\bCE[- ]mark",
    # guideline and professional bodies
    "NICE": r"\bNICE\b(?! (?:to|and|way|thing|idea|job|work|list|option|touch|feature|bonus|little|if))",
    "NHS": r"\bNHS\b",
    "American Psychiatric Association": r"american psychiatric association",
    "American Psychological Association": r"american psychological association|division 12|society of clinical psychology",
    "APA (bare acronym)": r"\bAPA\b",
    "WHO": r"world health organi[sz]ation|\bWHO\b(?=[^a-z]{0,3}(?:recommend|guideline|mhgap))",
    "NIMH / NIH": r"\bNIMH\b|national institute of mental health|\bNIH\b",
    "SAMHSA": r"\bSAMHSA\b",
    "VA / DoD": r"\bVA\b(?=[^a-z]{0,3}(?:/DoD|guideline|national center|app))|department of veterans affairs|national center for ptsd|\bDoD\b",
    "CANMAT": r"\bCANMAT\b",
    "RANZCP": r"\bRANZCP\b",
    "BAP": r"british association for psychopharmacology",
    "AACAP": r"\bAACAP\b",
    "AASM": r"\bAASM\b|american academy of sleep medicine",
    "AAP": r"american academy of pediatrics|\bAAP\b",
    # app review bodies
    "ORCHA / NHS app library / app reviewers": r"\bORCHA\b|nhs apps? library|app advisor|psyberguide|one mind|mind ?apps",
    # evidence sources and nosologies
    "Cochrane": r"cochrane",
    "peer-reviewed / trials": r"peer[- ]reviewed|clinical trials?|\bRCTs?\b|randomi[sz]ed",
    "meta-analysis": r"meta-analy",
    "DSM / ICD": r"\bDSM(?:-5|-IV|-5-TR)?\b|\bICD-1[01]\b",
    "HIPAA / GDPR": r"\bHIPAA\b|\bGDPR\b",
    # justification phrases
    "'evidence-based'": r"evidence[- ]based",
    "'clinically validated'": r"clinically (?:validated|proven|tested|studied)",
    "'gold standard'": r"gold[- ]standard",
    "'first-line'": r"first[- ]line",
    "'guideline'": r"guideline",
    "'approved'": r"\bapproved\b",
    "'cleared'": r"\bcleared\b",
    "'off-label'": r"off[- ]label",
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", "--dev", dest="runs"); args = ap.parse_args()
    runs_dir = runs_root(args.runs)
    pats = {k: re.compile(v, re.I) for k, v in REFERENCES.items()}
    hits = defaultdict(Counter); denom = Counter()
    fda_next = Counter()
    for r in iter_corpus(runs_dir):
        keys = ("all", f"footing|{r['footing']}", f"ask|{r['ask']}", f"family|{FAMILY[r['model']]}", f"model|{r['model']}")
        for k in keys:
            denom[k] += 1
        t = r["text"]
        for name, p in pats.items():
            if p.search(t):
                for k in keys:
                    hits[k][name] += 1
        for m in re.finditer(r"\bFDA\b[- ]?(\w+)", t):
            fda_next[m.group(1).lower()] += 1
    out = {"note": "share of responses containing a whole-word match; a match is a mention, not a correct use",
           "patterns": REFERENCES,
           "shares": {k: {name: hits[k][name] / denom[k] for name in REFERENCES} for k in denom},
           "n": dict(denom),
           "word_after_fda": dict(fda_next.most_common(10))}
    (DATA / "authorities.json").write_text(json.dumps(out, indent=1))
    s = out["shares"]
    print(f"{'reference':40s} {'all':>6s} | " + " ".join(f"{f[:4]:>6s}" for f in FOOTINGS) + " | " + " ".join(f"{a[:6]:>7s}" for a in ASKS))
    for name in REFERENCES:
        print(f"{name:40s} {100 * s['all'][name]:5.1f}% | " + " ".join(f"{100 * s[f'footing|{f}'][name]:5.1f}%" for f in FOOTINGS)
              + " | " + " ".join(f"{100 * s[f'ask|{a}'][name]:6.1f}%" for a in ASKS))
    print("word after FDA:", out["word_after_fda"])


if __name__ == "__main__":
    main()
