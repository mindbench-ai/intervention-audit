"""cells.json + concordance.json + responses.jsonl + entities.jsonl + ratings.jsonl ->
data/highlights.json: the numbers for selected cells and option groups, each under a
stable key, so that text quoting a number can cite the key rather than a hand-copied figure.

Groups: convergent cells, controls, the floor items (discredited practices), named
options rated below zero, and named single cells (DBS for OCD, lisdexamfetamine for binge
eating, complicated grief therapy for bereavement, and so on).

Each named cell also carries "per_model": for all 41 generation models, the model's own
recommendation rate for the option in that cell by footing and overall, its rate of
advising against it, and its own mean rating of it where the model was in the rating arm.

    python3 analysis/highlights.py
"""
import json
from collections import defaultdict

from common import CONTROLS, DATA, FAMILY, FOOTINGS, INSTRUMENT, LIST_ASK, MODELS, NULLS, read_jsonl

CELLS_OF_INTEREST = {
    "dbs_ocd": ("ocd|neuromodulation", "deep brain stimulation"),
    "rtms_ocd": ("ocd|neuromodulation", "repetitive transcranial magnetic stimulation"),
    "capsulotomy_ocd": ("ocd|neuromodulation", "anterior capsulotomy"),
    "rtms_mdd": ("major_depressive_disorder|neuromodulation", "repetitive transcranial magnetic stimulation"),
    "ect_mdd": ("major_depressive_disorder|neuromodulation", "electroconvulsive therapy"),
    "rtms_autism": ("autism_spectrum_disorder|neuromodulation", "repetitive transcranial magnetic stimulation"),
    "ect_bipolar": ("bipolar_1_maintenance|neuromodulation", "electroconvulsive therapy"),
    "lisdexamfetamine_bed": ("binge_eating_disorder|pharmacological", "lisdexamfetamine"),
    "complicated_grief_bereavement": ("uncomplicated_bereavement|psychotherapy", "complicated grief therapy"),
    "supportive_bereavement": ("uncomplicated_bereavement|psychotherapy", "supportive psychotherapy"),
    "cbti_insomnia": ("chronic_insomnia|psychotherapy", "cbt for insomnia"),
    "dbt_bpd": ("borderline_personality_disorder|psychotherapy", "dialectical behavior therapy"),
    "aripiprazole_ocd": ("ocd|pharmacological", "aripiprazole"),
    "risperidone_ocd": ("ocd|pharmacological", "risperidone"),
    "methamphetamine_adhd": ("adhd_adult|pharmacological", "methamphetamine"),
}


def per_model_cells():
    """{cell name -> {model -> per-model row}} for every entry in CELLS_OF_INTEREST.

    A cell is one condition and one list; its responses are the two asks that feed it,
    "open" and the list's own ask, so each model contributes 6 responses per footing (2
    asks x 3 samples) and 18 in all. named_by_footing is the share of those responses in
    which the model names the option without advising against it (entity rows deduplicated
    by response id, since one response can mention an option more than once);
    named_by_ask_footing splits the same by ask (3 responses per model, footing, and ask).
    advised_rate is the same over the advised-against rows. rating_mean is the model's own
    mean score for the option in the cell in the structured arm, None if it was not rated.
    """
    keyed = {}                       # (condition, list, option) -> cell name
    by_cond = defaultdict(list)      # condition -> [(cell name, list)]
    for name, (ck, opt) in CELLS_OF_INTEREST.items():
        cond, lst = ck.split("|")
        keyed[(cond, lst, opt)] = name
        by_cond[cond].append((name, lst))

    n_resp = defaultdict(int)                    # (cell, model, footing) -> responses
    n_resp_ask = defaultdict(int)                # (cell, model, footing, ask) -> responses
    for r in read_jsonl(DATA / "responses.jsonl"):
        for name, lst in by_cond.get(r["condition"], ()):
            if r["ask"] in ("open", LIST_ASK[lst]):
                n_resp[(name, r["model"], r["footing"])] += 1
                n_resp_ask[(name, r["model"], r["footing"], r["ask"])] += 1

    named = defaultdict(set)                     # (cell, model, footing) -> {rid}
    advised = defaultdict(set)
    named_ask = defaultdict(set)                 # (cell, model, footing, ask) -> {rid}
    for e in read_jsonl(DATA / "entities.jsonl"):
        if not e["option"]:
            continue
        name = keyed.get((e["condition"], e["list"], e["option"]))
        if name is None or e["ask"] not in ("open", LIST_ASK[e["list"]]):
            continue
        tgt = named if e["binary"] == "recommended" else advised
        tgt[(name, e["model"], e["footing"])].add(e["rid"])
        if e["binary"] == "recommended":
            named_ask[(name, e["model"], e["footing"], e["ask"])].add(e["rid"])

    rsum = defaultdict(float); rn = defaultdict(int)   # (cell, model) -> score sum / count
    for r in read_jsonl(DATA / "ratings.jsonl"):
        name = keyed.get((r["condition"], r["list"], r["option"]))
        if name is not None:
            rsum[(name, r["model"])] += r["score"]; rn[(name, r["model"])] += 1

    out = {}
    for name in CELLS_OF_INTEREST:
        rows = {}
        for m in MODELS:
            hits = adv = tot = 0
            by_f = {}
            for f in FOOTINGS:
                d = n_resp[(name, m, f)]
                by_f[f] = len(named[(name, m, f)]) / d if d else None
                hits += len(named[(name, m, f)]); adv += len(advised[(name, m, f)]); tot += d
            lst = CELLS_OF_INTEREST[name][0].split("|")[1]
            by_ask = {a: {f: (len(named_ask[(name, m, f, a)]) / n_resp_ask[(name, m, f, a)] if n_resp_ask[(name, m, f, a)] else None)
                          for f in FOOTINGS} for a in ("open", LIST_ASK[lst])}
            rows[m] = {"family": FAMILY[m], "named_by_footing": by_f, "named_by_ask_footing": by_ask,
                       "named_rate": hits / tot if tot else None,
                       "advised_rate": adv / tot if tot else None,
                       "rating_mean": rsum[(name, m)] / rn[(name, m)] if rn[(name, m)] else None,
                       "rating_n": rn[(name, m)]}
        out[name] = rows
    return out


def floor_items():
    out = []
    for lst, fn in (("psychotherapy", "talk_therapies.json"), ("neuromodulation", "neuromodulation.json")):
        for k, v in json.loads((INSTRUMENT / "lexicon" / fn).read_text()).items():
            if isinstance(v, dict) and v.get("floor"):
                out.append((lst, v["name"].lower(), v["name"]))
    return out


def main():
    cells = json.load((DATA / "cells.json").open())
    conc = json.load((DATA / "concordance.json").open())
    hl = {"cells": {}, "controls": {}, "floor": {}, "named_rated_below_zero": conc["exceptions"][:60]}

    def pick(cellkey, opt):
        cell = cells[cellkey]["options"]
        if opt not in cell:
            cands = [k for k in cell if opt.split()[0] in k]
            raise SystemExit(f"{opt!r} not in {cellkey}; candidates: {cands[:5]}")
        v = cell[opt]
        return {"display": v["display"], "named_rate": v["named_rate"], "named_by_footing": v["named_by_footing"],
                "advised_rate": v["advised_rate"], "advised_by_footing": v["advised_by_footing"],
                "n_models_named": v["n_models_named"], "n_models_advised": v["n_models_advised"],
                "rating_mean": v["rating_mean"], "rating_by_footing": v["rating_by_footing"], "rating_hist": v["rating_hist"]}

    pm = per_model_cells()
    for name, (ck, opt) in CELLS_OF_INTEREST.items():
        hl["cells"][name] = {"cell": ck, **pick(ck, opt), "per_model": pm[name]}
        # every model contributes the same number of responses, so the mean over models of
        # the per-model rates must reproduce the cell-level rate
        for f in FOOTINGS:
            got = sum(pm[name][m]["named_by_footing"][f] for m in MODELS) / len(MODELS)
            want = hl["cells"][name]["named_by_footing"][f]
            assert abs(got - want) < 1e-9, (name, f, got, want)
        got = sum(pm[name][m]["named_rate"] for m in MODELS) / len(MODELS)
        assert abs(got - hl["cells"][name]["named_rate"]) < 1e-9, (name, got)

    # controls: null rating, null ranked first, top named options per list
    for c in CONTROLS:
        d = {}
        for lst in ("pharmacological", "psychotherapy", "neuromodulation", "digital"):
            cell = cells[f"{c}|{lst}"]["options"]
            top = max(((o, v) for o, v in cell.items() if not v["is_null"]), key=lambda kv: kv[1]["named_rate"])
            d[lst] = {"null_rating_mean": cell[NULLS[lst]]["rating_mean"], "top_named": top[1]["display"],
                      "top_named_rate": top[1]["named_rate"], "top_named_rating": top[1]["rating_mean"],
                      "n_options_named_5pct": sum(1 for v in cell.values() if not v["is_null"] and v["named_rate"] >= 0.05)}
        hl["controls"][c] = d
    # null ranked first across all control rating calls
    calls = defaultdict(dict)
    for r in read_jsonl(DATA / "ratings.jsonl"):
        if r["condition"] in CONTROLS:
            calls[(r["model"], r["condition"], r["footing"], r["list"], r["sample"])][r["option"]] = r["score"]
    first = defaultdict(lambda: [0, 0])
    for (m, c, f, lst, s), sc in calls.items():
        nulls = [v for k, v in sc.items() if k.startswith("no ")]
        if nulls:
            first[c][1] += 1; first[c][0] += nulls[0] >= max(sc.values())
    for c in CONTROLS:
        hl["controls"][c]["null_ranked_first_share"] = first[c][0] / first[c][1]

    # floor items: ratings >= 0 and > 0, and any naming as recommended
    items = floor_items()
    fl = {}
    for lst, opt, disp in items:
        n = ge0 = gt0 = 0; named_cells = []
        for key, cell in cells.items():
            if key.startswith("_") or not key.endswith("|" + lst):
                continue
            v = cell["options"].get(opt)
            if not v or not v["rating_hist"]:
                continue
            h = v["rating_hist"]; n += sum(h); ge0 += sum(h[3:]); gt0 += sum(h[4:])
            if v["named_rate"] > 0:
                named_cells.append({"cell": key, "named_rate": round(v["named_rate"], 4), "n_models": v["n_models_named"]})
        fl[disp] = {"list": lst, "n_ratings": n, "rated_ge_0": ge0, "rated_gt_0": gt0, "named_as_recommended_in": named_cells}
    hl["floor"] = {"items": fl, "total_ratings": sum(v["n_ratings"] for v in fl.values()),
                   "total_rated_gt_0": sum(v["rated_gt_0"] for v in fl.values()),
                   "never_ge_0": [k for k, v in fl.items() if v["rated_ge_0"] == 0]}
    (DATA / "highlights.json").write_text(json.dumps(hl, indent=1))
    for k, v in hl["cells"].items():
        by_f = " / ".join("%.0f%%" % (100 * v["named_by_footing"][f]) for f in ("clinical", "caregiver", "patient"))
        print(f"{k:32s} named {v['named_rate']:.1%} ({by_f}) adv {v['advised_rate']:.1%} models {v['n_models_named']} rating {v['rating_mean']:+.2f}")
    for c, d in hl["controls"].items():
        print(f"{c}: null ranked first {d['null_ranked_first_share']:.1%}; " + "; ".join(f"{l[:5]} null {d[l]['null_rating_mean']:+.2f} top {d[l]['top_named']} {d[l]['top_named_rate']:.0%}" for l in ('pharmacological','psychotherapy','neuromodulation','digital')))
    f = hl["floor"]; print(f"floor: {f['total_rated_gt_0']}/{f['total_ratings']} ratings > 0 ({f['total_rated_gt_0']/f['total_ratings']:.2%}); never >= 0: {f['never_ge_0']}")

    # per-model detail for the cell the lisdexamfetamine figure draws
    key = "lisdexamfetamine_bed"
    print(f"\nper model, {key} ({hl['cells'][key]['cell']}, {hl['cells'][key]['display']}):")
    print(f"{'model':<24} {'developer':<10} {'clin':>6} {'care':>6} {'pat':>6} {'all':>6} {'rating':>7} {'n':>4}")
    print("-" * 74)
    for m, v in hl["cells"][key]["per_model"].items():
        b = v["named_by_footing"]
        r = f"{v['rating_mean']:+.2f}" if v["rating_mean"] is not None else "-"
        print(f"{m:<24} {v['family']:<10} " + " ".join(f"{100 * b[x]:5.0f}%" for x in FOOTINGS)
              + f" {100 * v['named_rate']:5.0f}% {r:>7} {v['rating_n']:4d}")


if __name__ == "__main__":
    main()
