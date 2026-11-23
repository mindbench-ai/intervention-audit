"""Judgments -> two base tables.

data/entities.jsonl   one row per (response, extracted entity): rid, model, family,
                      language, pack_version, condition, footing, ask, sample, surface,
                      list + option it maps to (or null), four-way stance, binary stance.
data/responses.jsonl  one row per response: rid, model, family, language, pack_version,
                      condition, footing, ask, sample, n_entities, n_recommended, n_advised_against,
                      n_recommended_listed (mapped to a rating list), consult_professional,
                      not_a_doctor.

Judge output is the frozen corpus pass (35,055 responses). Where one response names the
same option under two surfaces, entities.jsonl keeps both rows (they are distinct
mentions); per-option rates downstream count a response once per option.

    python3 analysis/canonicalise.py [--runs <folder with runs/>]
"""
import argparse
from collections import Counter

from common import DATA, FAMILY, LANGUAGE, binary_stance, runs_root, iter_judgments, map_surface, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", "--dev", dest="runs")
    args = ap.parse_args()
    runs_dir = runs_root(args.runs)
    entities, responses = [], []
    stance_counts = Counter()
    for r in iter_judgments(runs_dir):
        an = r["annotation"]
        rid = f"{r['model_key']}|{r['item_key']}"
        base = {"rid": rid, "model": r["model_key"], "family": FAMILY.get(r["model_key"]),
                "language": LANGUAGE, "pack_version": r.get("pack_version"), "condition": r["condition"], "footing": r["register"], "ask": r["category"], "sample": r["sample"]}
        n_rec = n_adv = n_rec_listed = 0
        for e in an.get("entities") or []:
            surface = (e.get("surface") or "").strip()
            stance = e.get("stance")
            if not surface or stance not in ("recommended", "neutral", "advised_against", "mixed"):
                continue
            stance_counts[stance] += 1
            hits = sorted(map_surface(surface))
            b = binary_stance(stance)
            row = {**base, "surface": surface, "list": None, "option": None, "stance": stance, "binary": b}
            if hits:
                row["list"], row["option"] = hits[0]
                if len(hits) > 1:
                    row["also_maps_to"] = [f"{l}:{o}" for l, o in hits[1:]]
            entities.append(row)
            n_rec += b == "recommended"
            n_adv += b == "advised_against"
            n_rec_listed += b == "recommended" and bool(hits)
        responses.append({**base, "n_entities": n_rec + n_adv, "n_recommended": n_rec,
                          "n_advised_against": n_adv, "n_recommended_listed": n_rec_listed,
                          "consult_professional": bool(an.get("consult_professional")),
                          "not_a_doctor": bool(an.get("not_a_doctor"))})
    n_e = write_jsonl(DATA / "entities.jsonl", entities)
    n_r = write_jsonl(DATA / "responses.jsonl", responses)
    mapped = sum(1 for e in entities if e["option"])
    print(f"responses: {n_r} | entities: {n_e} | mapped to a list option: {mapped} ({mapped / n_e:.1%})")
    print("four-way stances:", {k: f"{v} ({v / n_e:.1%})" for k, v in stance_counts.most_common()})
    rec = sum(1 for e in entities if e["binary"] == "recommended")
    print(f"binary: recommended {rec} ({rec / n_e:.1%}), advised_against {n_e - rec} ({(n_e - rec) / n_e:.1%})")


if __name__ == "__main__":
    main()
