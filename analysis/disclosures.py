"""responses.jsonl -> data/disclosures.json: consult-a-professional and disclaimer rates.

Keys: "all", "footing|<f>", "ask|<a>", "footing_ask|<f>|<a>", "condition|<c>",
"ask_condition|<a>|<c>", "model|<m>", "model_footing|<m>|<f>". Each holds n (responses),
cp (share advising to consult a professional) and nad (share saying the model is not a
doctor or not qualified). "_footing_effect" holds, per model, the clinician rate minus the
lower of the two lay rates, and the model counts behind them.

    python3 analysis/disclosures.py
"""
import json
from collections import defaultdict

from common import DATA, read_jsonl


def main():
    agg = defaultdict(lambda: [0, 0, 0])
    for r in read_jsonl(DATA / "responses.jsonl"):
        cp, nad = int(r["consult_professional"]), int(r["not_a_doctor"])
        f, a, c, m = r["footing"], r["ask"], r["condition"], r["model"]
        for key in ("all", f"footing|{f}", f"ask|{a}", f"footing_ask|{f}|{a}", f"condition|{c}",
                    f"ask_condition|{a}|{c}", f"model|{m}", f"model_footing|{m}|{f}"):
            x = agg[key]; x[0] += 1; x[1] += cp; x[2] += nad
    out = {k: {"n": n, "cp": cp / n, "nad": nad / n} for k, (n, cp, nad) in agg.items()}
    # footing effect per model: clinician rate minus the lower of the two lay footings
    models = sorted({k.split("|")[1] for k in out if k.startswith("model|")})
    effect = {}
    for m in models:
        f = {ft: out[f"model_footing|{m}|{ft}"] for ft in ("clinical", "caregiver", "patient")}
        effect[m] = {"cp_clinical_minus_lay_min": f["clinical"]["cp"] - min(f["caregiver"]["cp"], f["patient"]["cp"]),
                     "nad_clinical_minus_lay_min": f["clinical"]["nad"] - min(f["caregiver"]["nad"], f["patient"]["nad"])}
    out["_footing_effect"] = {"per_model": effect,
                              "n_models": len(models),
                              "n_models_consult_lower_for_clinician": sum(1 for v in effect.values() if v["cp_clinical_minus_lay_min"] < 0),
                              "n_models_notdoc_lower_for_clinician": sum(1 for v in effect.values() if v["nad_clinical_minus_lay_min"] < 0),
                              "n_models_notdoc_under_2pct": sum(1 for m in models if out[f"model|{m}"]["nad"] < 0.02)}
    (DATA / "disclosures.json").write_text(json.dumps(out, indent=1))
    fe = out["_footing_effect"]
    print(f"footing effect: consult lower for clinician in {fe['n_models_consult_lower_for_clinician']}/{fe['n_models']} models; not-a-doctor lower in {fe['n_models_notdoc_lower_for_clinician']}; models with not-a-doctor < 2%: {fe['n_models_notdoc_under_2pct']}")
    a = out["all"]; print(f"responses {a['n']}: consult {a['cp']:.1%}, not-a-doctor {a['nad']:.1%}")
    for f in ("clinical", "caregiver", "patient"):
        x = out[f"footing|{f}"]; print(f"  {f:10s} consult {x['cp']:.1%} | not-a-doctor {x['nad']:.1%}")


if __name__ == "__main__":
    main()
