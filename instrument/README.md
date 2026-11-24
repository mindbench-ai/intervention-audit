# Instrument

The instrument is everything the models are asked and everything their answers are matched
against. It is versioned and frozen before each run. `VERSION.json` lists the vignette, ask,
option-list and review-sheet files with their hashes, the runners stamp the version on every row they write, and `check_pack.py`
checks that the frozen files are intact and that each one still rebuilds from its
expert-review sheet.

    python3 instrument/check_pack.py

## Files

| File | Contents |
|---|---|
| `dataset/situations.json` | The 19 case vignettes, 17 conditions and 2 controls (listed under `_controls`), each written in the three footings |
| `dataset/asks.json` | The five asks (open, medication, psychotherapy, neuromodulation, digital), each in the three footings. The `_deferred` block holds two asks that were not used in any run |
| `lexicon/psych_meds.json` | Medication list: 139 entries, each with drug class, brand names and aliases |
| `lexicon/talk_therapies.json` | Psychotherapy list: 70 entries with family, abbreviation and aliases, including 8 floor items and the "no psychotherapy" option |
| `lexicon/neuromodulation.json` | Procedures list: 35 entries with family, invasiveness tier and aliases, including 6 floor items and the "no somatic or neuromodulation treatment" option |
| `lexicon/digital.json` | Digital list: 102 apps and AI tools with a regulatory tag, availability and aliases, including the "no digital intervention" option |
| `lexicon/anchors.json` | Wording of the rating task: the instruction template, the seven scale anchors, and the null option shown with each list ("no medication" is added here). It is not listed in `VERSION.json` |
| `expert_review/` | The reviewed sheets the files above are built from, the instructions the reviewers received, and the style rules for the footings |
| `VERSION.json` | The version, the freeze date and the hash of every file |
| `check_pack.py` | The integrity and rebuild check above |

The tags on the option lists (drug class, family, invasiveness tier, regulatory tag,
availability) are not shown to the models, nor are they used in analysis in v1.0.0, apart
from the family value that marks each list's null option. They may be used in future
versions of the audit pending additional clinical validation. Floor items are discredited
or historical practices carried on a list on purpose, so that a model recommending one is a
finding.

## Versions

Version 1.0.0 was built from the expert-review sheets and frozen on 2026-08-21. The
versioning policy is in the top-level README, and `CHANGELOG.md` records each version. A
new version is a new set of files with a new `VERSION.json`; the `dataset/` and `lexicon/`
files of a version are never edited after it is frozen.
