"""Judge comparison on the frozen test split (pre-specified; run once on 2026-09-08).

Units = (response, judge entity) coded by both assigned humans. Judge stance from the
frozen corpus pass (runs/judge_v1/luna/judgments.jsonl in the raw runs). Human
codings from validation/codings/. Reports human-human vs judge-human agreement on the pre-specified endpoints,
checkbox agreement, extraction precision (human 'not a treatment' rejections) and
recall proxy (human free-lists), and the corpus-reweighted aggregate.
"""
import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ag", HERE / "agreement_humans.py")
ag = importlib.util.module_from_spec(spec); spec.loader.exec_module(ag)
alpha, kappa = ag.alpha, ag.kappa

CORPUS_SHARE = {"recommended": .853, "advised_against": .088, "neutral": .043, "mixed": .016}


JUDGMENTS = ROOT / "runs/judge_v1/luna/judgments.jsonl"  # overridden by --runs


def load_judge(rids):
    out = {}
    for line in JUDGMENTS.open():
        r = json.loads(line)
        rid = f'{r["model_key"]}|{r["item_key"]}'
        if rid not in rids:
            continue
        an = r.get("annotation") or {}
        st = {}
        for e in an.get("entities") or []:
            st.setdefault(e["surface"], e["stance"])
        out[rid] = {"stances": st, "not_a_doctor": bool(an.get("not_a_doctor")),
                    "consult_professional": bool(an.get("consult_professional"))}
    return out


def fold(v): return "recommended" if v == "mixed" else v
def binary(v): return "adv" if v == "advised_against" else "not"


def run(split):
    units, resp = ag.units_for(2, split)
    judge = load_judge(set(resp))
    res = {"n_responses": len(resp), "n_judge_rows": len(judge)}
    hh, jh, unan = [], [], []
    na_any = na_both = n_units = 0
    per_cat = defaultdict(lambda: [0, 0])          # judge category -> [n human labels, n agreeing]
    for (rid, e), u in units.items():
        if len(u) != 2 or rid not in judge:
            continue
        hv = list(u.values())
        if not all(hv):
            continue
        jv = judge[rid]["stances"].get(e)
        if jv is None:
            continue
        n_units += 1
        nas = [v == "not_applicable" for v in hv]
        na_any += any(nas); na_both += all(nas)
        if any(nas):
            continue
        hh.append(tuple(hv))
        for v in hv:
            jh.append((jv, v))
            per_cat[jv][0] += 1; per_cat[jv][1] += (v == jv)
        if hv[0] == hv[1]:
            unan.append((jv, hv[0]))
    res["n_units_dual_coded"] = n_units
    res["extraction_precision"] = {"rejected_by_either": na_any / n_units, "rejected_by_both": na_both / n_units,
                                   "precision_lower": 1 - na_any / n_units, "precision_upper": 1 - na_both / n_units}
    def block(pairs):
        return {"four_way": alpha(pairs), "folded": alpha([(fold(a), fold(b)) for a, b in pairs]),
                "binary_advised_against": alpha([(binary(a), binary(b)) for a, b in pairs]),
                "ordinal": alpha([(fold(a), fold(b)) for a, b in pairs], "ordinal"),
                "excl_neutral_units": alpha([p for p in pairs if "neutral" not in p]),
                "n": len(pairs)}
    res["human_human"] = block(hh)
    res["judge_human"] = block(jh)
    res["judge_vs_unanimous"] = {"n": len(unan), "match": sum(a == b for a, b in unan) / len(unan),
                                 "match_binary": sum(binary(a) == binary(b) for a, b in unan) / len(unan),
                                 "by_human_label": {c: {"n": n, "match": m / n} for c, (n, m) in
                                                    {c: (sum(1 for a, b in unan if b == c), sum(1 for a, b in unan if b == c and a == b)) for c in set(b for _, b in unan)}.items()}}
    res["judge_category_agreement"] = {c: {"n": v[0], "human_agrees": v[1] / v[0]} for c, v in per_cat.items()}
    # corpus-reweighted aggregate: weight per-judge-category agreement by corpus stance share
    res["judge_human_agreement_sample"] = sum(v[1] for v in per_cat.values()) / sum(v[0] for v in per_cat.values())
    res["judge_human_agreement_corpus_weighted"] = sum(CORPUS_SHARE[c] * per_cat[c][1] / per_cat[c][0] for c in CORPUS_SHARE if per_cat[c][0])
    # judge stance distribution in sample vs corpus
    res["judge_stance_share_sample"] = {c: per_cat[c][0] / sum(v[0] for v in per_cat.values()) for c in per_cat}
    # checkboxes
    for key in ("consult_professional", "not_a_doctor"):
        hhp, jhp, un = [], [], []
        for rid, v in resp.items():
            if len(v) != 2 or rid not in judge: continue
            a, b = (r[key] for r in v.values()); j = judge[rid][key]
            hhp.append((a, b)); jhp += [(j, a), (j, b)]
            if a == b: un.append((j, a))
        yn = lambda v: "y" if v else "n"  # noqa: E731
        res[key] = {"human_human_kappa": kappa(hhp)[0], "human_human_agree": kappa(hhp)[1],
                    "judge_human_kappa": kappa(jhp)[0], "judge_human_agree": kappa(jhp)[1],
                    "human_human_alpha": alpha([(yn(a), yn(b)) for a, b in hhp]),
                    "judge_human_alpha": alpha([(yn(a), yn(b)) for a, b in jhp]),
                    "judge_vs_unanimous_agree": sum(a == b for a, b in un) / len(un), "n_unanimous": len(un),
                    "rate_judge": sum(j for j, _ in jhp) / len(jhp), "rate_human": sum(a for _, a in jhp) / len(jhp)}
    # recall proxy from free-lists
    n_items = 0; n_cod = 0; n_with = 0
    for rid, v in resp.items():
        for r in v.values():
            n_cod += 1
            m = r["missed"].strip()
            if m:
                n_with += 1
                n_items += len([x for x in re.split(r"[\n;,]+|\s+and\s+", m) if x.strip()])
    n_judge_ents = sum(len(judge[rid]["stances"]) for rid in resp if rid in judge)
    res["recall_proxy"] = {"codings_with_free_list": n_with / n_cod, "free_list_items_total": n_items,
                           "judge_entities_total": n_judge_ents,
                           "recall_lower_bound": n_judge_ents / (n_judge_ents + n_items / 2)}
    return res


def main():
    global JUDGMENTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", "--dev", dest="runs", help="folder that holds runs/judge_v1 (default: this repository)")
    args = ap.parse_args()
    if args.runs:
        JUDGMENTS = Path(args.runs) / "runs/judge_v1/luna/judgments.jsonl"
    out = {s: run(s) for s in ("test", "dev")}
    (HERE / "judge_comparison.json").write_text(json.dumps(out, indent=1))
    for s in ("test", "dev"):
        r = out[s]
        print(f"\n===== {s.upper()} split: {r['n_responses']} responses, {r['n_units_dual_coded']} dual-coded judge entities")
        print(f"{'':30s}{'human-human':>14s}{'judge-human':>14s}")
        for k in ("binary_advised_against", "ordinal", "folded", "four_way", "excl_neutral_units"):
            print(f"{k:30s}{r['human_human'][k]:14.3f}{r['judge_human'][k]:14.3f}")
        u = r["judge_vs_unanimous"]
        print(f"judge matches unanimous humans: {u['match']:.3f} (binary {u['match_binary']:.3f}) on n={u['n']}; by label:",
              {c: (v['n'], round(v['match'], 2)) for c, v in u['by_human_label'].items()})
        print("judge category -> human agrees:", {c: (v['n'], round(v['human_agrees'], 2)) for c, v in r["judge_category_agreement"].items()})
        print(f"judge-human agreement: sample {r['judge_human_agreement_sample']:.3f}, corpus-reweighted {r['judge_human_agreement_corpus_weighted']:.3f}")
        p = r["extraction_precision"]
        print(f"extraction precision: {p['precision_lower']:.3f} (either rejects) .. {p['precision_upper']:.3f} (both reject)")
        for key in ("consult_professional", "not_a_doctor"):
            c = r[key]
            print(f"{key}: HH kappa {c['human_human_kappa']:.2f} ({c['human_human_agree']:.2f}) | JH kappa {c['judge_human_kappa']:.2f} ({c['judge_human_agree']:.2f}) | judge vs unanimous {c['judge_vs_unanimous_agree']:.2f} n={c['n_unanimous']} | rate judge {c['rate_judge']:.2f} human {c['rate_human']:.2f}")
        rc = r["recall_proxy"]
        print(f"recall proxy: free-list on {rc['codings_with_free_list']:.0%} of codings, {rc['free_list_items_total']} items vs {rc['judge_entities_total']} judge entities -> recall ~{rc['recall_lower_bound']:.3f}")


if __name__ == "__main__":
    main()
