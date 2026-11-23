"""Human-human agreement on the main wave, from the release codings. The judge is NOT
loaded here; this is the human-only investigation that precedes the one-shot judge
comparison on the frozen test split.

Reads validation/codings/codings_all.jsonl (main-wave rounds only).
Writes validation/agreement_humans.json and prints a report.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (HERE / "codings/codings_all.jsonl").open() if json.loads(l)["round"] in (1, 2)]

ORD = {"advised_against": 0, "neutral": 1, "recommended": 2}


def alpha(pairs, metric="nominal"):
    """Krippendorff's alpha for units with exactly 2 coders (pairs of labels)."""
    pairs = [(a, b) for a, b in pairs if a and b]
    if len(pairs) < 2:
        return None
    n = Counter(); o = Counter()
    for a, b in pairs:
        n[a] += 1; n[b] += 1
        o[(a, b)] += 1; o[(b, a)] += 1
    cats = sorted(n)
    tot = sum(n.values())
    if metric == "nominal":
        d = lambda c, k: 0.0 if c == k else 1.0
    else:
        cs = sorted(cats, key=lambda c: ORD[c]); idx = {c: i for i, c in enumerate(cs)}
        def d(c, k):
            if c == k: return 0.0
            lo, hi = sorted((idx[c], idx[k]))
            s = sum(n[cs[g]] for g in range(lo, hi + 1)) - (n[c] + n[k]) / 2
            return s * s
    Do = sum(v * d(c, k) for (c, k), v in o.items())
    De = sum(n[c] * n[k] * d(c, k) for c in cats for k in cats) / (tot - 1)
    return None if De == 0 else 1 - Do / De


def kappa(pairs):
    pairs = [(bool(a), bool(b)) for a, b in pairs]
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n; pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return ((po - pe) / (1 - pe) if pe < 1 else None), po


def units_for(round_no, split):
    by = defaultdict(dict); resp = defaultdict(dict)
    for r in rows:
        if r["round"] != round_no or r["split"] != split:
            continue
        resp[r["rid"]][r["coder"]] = r
        for e, v in r["stances"].items():
            by[(r["rid"], e)][r["coder"]] = v
    return by, resp


def report_split(round_no, split, out):
    units, resp = units_for(round_no, split)
    pairs = [tuple(u.values()) for u in units.values() if len(u) == 2]
    res = {"n_responses": len(resp), "n_units": len(pairs),
           "unanswered_cells": sum(1 for p in pairs for v in p if not v)}
    fold = lambda v: "recommended" if v == "mixed" else v
    p_na = [p for p in pairs if all(p)]
    p4 = [p for p in p_na if "not_applicable" not in p]
    res["alpha_5way"] = alpha(p_na)
    res["alpha_4way_stance"] = alpha(p4)
    res["alpha_folded_mixed_to_rec"] = alpha([(fold(a), fold(b)) for a, b in p4])
    res["alpha_excl_neutral_units"] = alpha([p for p in p4 if "neutral" not in p])
    res["alpha_binary_advised_against"] = alpha([("adv" if a == "advised_against" else "not",
                                                  "adv" if b == "advised_against" else "not") for a, b in p4])
    res["alpha_ordinal_3"] = alpha([(fold(a), fold(b)) for a, b in p4], metric="ordinal")
    cat = defaultdict(lambda: [0, 0])
    for a, b in p_na:
        for x, y in ((a, b), (b, a)):
            cat[x][0] += 1; cat[x][1] += (x == y)
    res["per_category"] = {c: {"n": v[0], "agree": v[1] / v[0]} for c, v in sorted(cat.items())}
    conf = Counter(tuple(sorted(p)) for p in p_na if p[0] != p[1])
    res["top_disagreements"] = [{"labels": list(k), "n": v} for k, v in conf.most_common(8)]
    for key in ("consult_professional", "not_a_doctor"):
        cp = [tuple(r[key] for r in v.values()) for v in resp.values() if len(v) == 2]
        k, po = kappa(cp)
        res[f"{key}_kappa"] = k; res[f"{key}_pct_agree"] = po
        res[f"{key}_rate"] = sum(a + b for a, b in cp) / (2 * len(cp))
    res["missed_nonempty_rate"] = sum(1 for v in resp.values() for r in v.values() if r["missed"].strip()) / sum(len(v) for v in resp.values())
    pair_units = defaultdict(list); coder_units = defaultdict(list)
    for (rid, e), u in units.items():
        if len(u) != 2: continue
        (c1, a), (c2, b) = sorted(u.items())
        if not a or not b or "not_applicable" in (a, b): continue
        pair_units[f"{c1}-{c2}"].append((a, b))
        coder_units[c1].append((a, b)); coder_units[c2].append((b, a))
    res["per_pair_alpha_4way"] = {k: {"n": len(v), "alpha": alpha(v)} for k, v in sorted(pair_units.items())}
    res["per_coder_alpha_4way"] = {k: {"n": len(v), "alpha": alpha(v)} for k, v in sorted(coder_units.items())}
    res["per_coder_label_use"] = {c: dict(Counter(a for a, _ in v)) for c, v in sorted(coder_units.items())}
    out[f"round{round_no}_{split}"] = res
    return res


def self_consistency(out):
    r1 = {(r["coder"], r["rid"]): r for r in rows if r["round"] == 1}
    r2 = {(r["coder"], r["rid"]): r for r in rows if r["round"] == 2 and r["split"] == "dev"}
    pairs = []; changed = Counter(); cb = {"consult_professional": [], "not_a_doctor": []}
    for k, a in r1.items():
        b = r2.get(k)
        if not b: continue
        for e, v in a["stances"].items():
            w = b["stances"].get(e)
            if v and w:
                pairs.append((v, w))
                if v != w: changed[(v, w)] += 1
        for key in cb: cb[key].append((a[key], b[key]))
    res = {"n_units": len(pairs), "pct_unchanged": sum(a == b for a, b in pairs) / len(pairs),
           "alpha_r1_vs_r2_same_coder": alpha(pairs),
           "top_changes": [{"from": k[0], "to": k[1], "n": v} for k, v in changed.most_common(8)]}
    for key, v in cb.items():
        res[f"{key}_changed"] = sum(a != b for a, b in v)
        res[f"{key}_rate_r1"] = sum(a for a, _ in v) / len(v)
        res[f"{key}_rate_r2"] = sum(b for _, b in v) / len(v)
    out["dev_self_consistency"] = res
    return res


def main():
    out = {}
    for round_no, split in ((1, "dev"), (2, "dev"), (2, "test")):
        r = report_split(round_no, split, out)
        print(f"\n== round {round_no}, {split} split: {r['n_responses']} responses, {r['n_units']} dual-coded entities, {r['unanswered_cells']} unanswered cells")
        for k in ("alpha_5way", "alpha_4way_stance", "alpha_folded_mixed_to_rec", "alpha_excl_neutral_units",
                  "alpha_binary_advised_against", "alpha_ordinal_3"):
            print(f"  {k:32s} {r[k]:.3f}" if r[k] is not None else f"  {k:32s} n/a")
        print("  per category (n, agree):", {c: (v['n'], round(v['agree'], 2)) for c, v in r["per_category"].items()})
        print("  top disagreements:", [(d['labels'], d['n']) for d in r["top_disagreements"][:5]])
        for key in ("consult_professional", "not_a_doctor"):
            print(f"  {key}: rate={r[key+'_rate']:.2f} pct_agree={r[key+'_pct_agree']:.2f} kappa={r[key+'_kappa']:.2f}")
        print(f"  free-list used in {r['missed_nonempty_rate']:.0%} of codings")
        if split == "test":
            print("  per coder 4-way alpha:", {c: round(v['alpha'], 2) for c, v in r["per_coder_alpha_4way"].items()})
            print("  per pair 4-way alpha:", {c: round(v['alpha'], 2) for c, v in r["per_pair_alpha_4way"].items() if v['alpha'] is not None})
            print("  per coder label use:", r["per_coder_label_use"])
    s = self_consistency(out)
    print(f"\n== dev re-review: same coder round 1 vs round 2: {s['n_units']} entities, {s['pct_unchanged']:.0%} unchanged, alpha {s['alpha_r1_vs_r2_same_coder']:.3f}")
    print("  top changes:", [(d['from'], d['to'], d['n']) for d in s['top_changes'][:6]])
    for key in ("consult_professional", "not_a_doctor"):
        print(f"  {key}: changed on {s[key+'_changed']} codings; rate r1 {s[key+'_rate_r1']:.2f} -> r2 {s[key+'_rate_r2']:.2f}")
    (HERE / "agreement_humans.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
