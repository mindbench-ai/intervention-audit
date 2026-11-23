"""Round-1 checkpoint: human-human agreement over the 100 development responses, each coded
by its two assigned coders, computed before round 2 was released. Reports pooled alpha
(five-way and with mixed folded into recommended), raw agreement, per-category agreement,
per-coder profiles, the top disagreement patterns, and the two checkbox questions.

Reads validation/codings/R<i>_round1.json, validation/assignments.json and the round-1
packets under validation/packets/main_wave/.

    python3 validation/checkpoint_round1.py
"""
import json, re, sys, html as H
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
meta = json.loads((HERE / 'assignments.json').read_text())
coders = {}
for i in range(1, 9):
    d = json.loads((HERE / f'codings/R{i}_round1.json').read_text())
    coders[d["coder"]] = {a["rid"]: a for a in d["answers"]}

valid_ents = {}
for i in range(1, 9):
    page = (HERE / f'packets/main_wave/R{i}_round1.html').read_text()
    for m in re.finditer(r'data-rid="([^"]+)" data-ents=\'([^\']*)\'', page):
        valid_ents[H.unescape(m.group(1))] = set(json.loads(H.unescape(m.group(2))))

units = {}   # (rid, ent) -> {coder: stance}
for rid, a in meta["assignments"].items():
    if a["wave"] != "1": continue
    c1, c2 = a["coders"]
    for e in valid_ents[rid]:
        units[(rid, e)] = {c1: coders[c1][rid]["stances"].get(e),
                           c2: coders[c2][rid]["stances"].get(e)}

def alpha(pairs):
    o, n = Counter(), Counter()
    for a, b in pairs:
        if not a or not b: continue
        n[a] += 1; n[b] += 1
        o[(a, b)] += 1; o[(b, a)] += 1
    tot = sum(n.values())
    Do = sum(v for (c, k), v in o.items() if c != k)
    De = sum(n[c] * n[k] / (tot - 1) for c in n for k in n if c != k)
    return 1 - Do / De

pairs_all = [tuple(u.values()) for u in units.values()]
fold = lambda v: 'recommended' if v == 'mixed' else v
print(f"CHECKPOINT: {len(units)} entity units over 100 dev responses, 8 coders")
print(f"pooled alpha (5-way): {alpha(pairs_all):.3f}")
print(f"pooled alpha (folded mixed->rec): {alpha([(fold(a), fold(b)) for a, b in pairs_all]):.3f}")
raw = sum(1 for a, b in pairs_all if a == b and a) / sum(1 for a, b in pairs_all if a and b)
print(f"raw agreement: {raw:.0%}")

print("\nPer-category (modal label): unanimous/total")
for cat in ("recommended", "neutral", "advised_against", "mixed", "not_applicable"):
    cu = [(a, b) for a, b in pairs_all if a and b and (a == cat or b == cat)]
    agree = sum(1 for a, b in cu if a == b)
    print(f"  {cat:16s} {agree}/{len(cu)}")

print("\nPer-coder profile (agreement with partners | na rate | mixed rate | neutral rate | avg s/resp):")
for c in sorted(coders):
    cu = [(u[c], [v for k, v in u.items() if k != c][0]) for u in units.values() if c in u]
    cu = [(a, b) for a, b in cu if a and b]
    agr = sum(1 for a, b in cu if a == b) / len(cu)
    my = [a for a, b in cu]
    times = [x["seconds"] for x in coders[c].values() if x.get("seconds") and x["seconds"] < 3600]
    avg = sum(times) / len(times) if times else None
    print(f"  {c}: {agr:.0%} | na {my.count('not_applicable')/len(my):.1%} | "
          f"mixed {my.count('mixed')/len(my):.1%} | neutral {my.count('neutral')/len(my):.1%} | "
          f"{round(avg) if avg else '?'}s ({len(cu)} units)")

conf = Counter()
for a, b in pairs_all:
    if a and b and a != b: conf[tuple(sorted((a, b)))] += 1
print("\nTop disagreement patterns:", conf.most_common(6))

dis_by_resp = Counter()
for (rid, e), u in units.items():
    vals = [v for v in u.values() if v]
    if len(set(vals)) > 1: dis_by_resp[rid] += 1
print(f"\nresponses with >=5 disagreeing units: "
      f"{[(r.split('|')[0] + '|' + r.split('|')[1][:18], n) for r, n in dis_by_resp.most_common(8) if n >= 5]}")

for field in ("not_a_doctor", "consult_professional"):
    tot = agree = 0
    for rid, a in meta["assignments"].items():
        if a["wave"] != "1": continue
        c1, c2 = a["coders"]
        tot += 1
        agree += coders[c1][rid][field] == coders[c2][rid][field]
    print(f"{field}: {agree}/{tot} agree")
