"""Calibration-pilot agreement report: three coders on the same ten responses.

Units are (response, highlighted entity) pairs from the pilot packet's entity lists.
Reports nominal Krippendorff's alpha over the five answers (recommended, neutral, advised
against, mixed, not a treatment), pairwise alpha and raw agreement, per-category
unanimity, every disagreement, the two checkbox questions, timing, and the free-lists.
The pre-specified gates were alpha >= .67 to proceed with instrument revisions and >= .80
to proceed as is.

Reads validation/codings/R<i>_<pilot>.json and validation/packets/<pilot packet>.

    python3 validation/pilot_agreement.py pilot1|pilot2
"""
import html as H
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATS = ["recommended", "neutral", "advised_against", "mixed", "not_applicable"]
PACKET = {"pilot1": "pilot_packet.html", "pilot2": "pilot2_packet.html"}


def alpha(units):
    o, n = Counter(), Counter()
    for vals in units:
        vals = [v for v in vals if v]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            n[vals[i]] += 1
            for j in range(m):
                if i != j:
                    o[(vals[i], vals[j])] += 1 / (m - 1)
    total = sum(n.values())
    if total < 2:
        return None
    Do = sum(v for (c, k), v in o.items() if c != k)
    De = sum(n[c] * n[k] / (total - 1) for c in n for k in n if c != k)
    return 1 - Do / De if De else None


def main():
    pilot = sys.argv[1] if len(sys.argv) > 1 else "pilot1"
    packet = (HERE / "packets" / PACKET[pilot]).read_text()
    valid = {H.unescape(m.group(1)): set(json.loads(H.unescape(m.group(2))))
             for m in re.finditer(r'data-rid="([^"]+)" data-ents=\'([^\']*)\'', packet)}
    files = sorted((HERE / "codings").glob(f"R*_{pilot}.json"))
    exports = {json.loads(f.read_text())["coder"]: json.loads(f.read_text()) for f in files}
    coders = {c: {a["rid"]: a for a in d["answers"]} for c, d in exports.items()}
    names = sorted(coders)
    print(f"{pilot}: coders {names}")

    units = defaultdict(dict)
    skipped = Counter()
    for cn, answers in coders.items():
        for rid, a in answers.items():
            for ent, st in a["stances"].items():
                if ent in valid.get(rid, set()):
                    units[(rid, ent)][cn] = st
            skipped[cn] += len(valid.get(rid, set()) - set(a["stances"]))
    print(f"highlights left unanswered: {dict(skipped)}")

    all_units = [[u.get(c) for c in names] for u in units.values()]
    n2 = len([v for v in all_units if sum(x is not None for x in v) >= 2])
    print(f"\nPOOLED alpha (5-way, {n2} units with >= 2 coders): {alpha(all_units):.3f}")
    fold = lambda v: "recommended" if v == "mixed" else v  # noqa: E731
    print(f"POOLED alpha (mixed folded into recommended): {alpha([[fold(x) if x else x for x in v] for v in all_units]):.3f}")
    for a, b in combinations(names, 2):
        pu = [[u.get(a), u.get(b)] for u in units.values() if u.get(a) and u.get(b)]
        raw = sum(1 for x, y in pu if x == y) / len(pu)
        print(f"  {a}-{b}: alpha {alpha(pu):.3f}, raw {raw:.0%} ({len(pu)} shared units)")

    print("\nPer-category (units whose modal label is the category: unanimous/total):")
    for cat in CATS:
        cu = [v for v in all_units if sum(x is not None for x in v) >= 2
              and Counter(x for x in v if x).most_common(1)[0][0] == cat]
        if cu:
            print(f"  {cat:16s} {sum(1 for v in cu if len(set(x for x in v if x)) == 1)}/{len(cu)}")

    print("\nDISAGREEMENTS (any two coders differ):")
    for (rid, ent), u in sorted(units.items()):
        vals = {c: u.get(c) for c in names if u.get(c)}
        if len(set(vals.values())) > 1:
            print(f"  {rid.split('|')[1][:22]:22s} {ent[:38]:38s} " + "  ".join(f"{c}:{v[:5]}" for c, v in vals.items()))

    print("\nCheckbox questions:")
    for field in ("not_a_doctor", "consult_professional"):
        agree = tot = 0
        for rid in valid:
            vals = [coders[c][rid][field] for c in names if rid in coders[c]]
            if len(vals) >= 2:
                tot += 1
                agree += len(set(vals)) == 1
        print(f"  {field}: {agree}/{tot} responses unanimous")

    print("\nTiming (avg s/resp):", {c: d["avg_seconds_per_response"] for c, d in exports.items()})
    print("\nFree-lists:")
    for c, d in exports.items():
        for a in d["answers"]:
            if a["missed"].strip():
                print(f"  {c} {a['slot']:14s}: {a['missed']}")


if __name__ == "__main__":
    main()
