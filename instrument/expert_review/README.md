# Expert-review sheets

These six sheets are the reviewed sources of the instrument's runtime files. `check_pack.py`
rebuilds each runtime file from the content columns of its sheet and checks the result
against the frozen file.

| Sheet | Runtime file | Reviewed by |
|---|---|---|
| `01d_patient_scenarios_v4.csv` | `dataset/situations.json` | Reviewer A |
| `02d_asks_v4.csv` | `dataset/asks.json` | Reviewer A |
| `05c_drugs_generic_v2.csv` | `lexicon/psych_meds.json` | Review group B |
| `08b_brain_stimulation_v2.csv` | `lexicon/neuromodulation.json` | Review group B |
| `09b_talk_therapies_v2.csv` | `lexicon/talk_therapies.json` | Review group C |
| `07b_digital_apps_v2.csv` | `lexicon/digital.json` | Review group D |

The note column of each sheet records what changed from the sheet's previous version and
why each added item is on the list. The style rules for the three footings of the vignettes
and asks are in `style-profiles-v3.md`.

The sheets here are copies of the frozen sheets with reviewer names replaced by these
letters and internal working notes removed. Only the note columns differ, so every runtime
file still rebuilds exactly, and `VERSION.json` records the hash of each sheet as frozen
and the hash of the copy published here.

## Instructions to reviewers

The instructions below were sent with the first review round in July 2026 and applied to
every later round; specific points that came up were discussed with the reviewers in
meetings. The first round covered earlier versions of these sheets (`01b` → `01d`, `02b` →
`02d`, `05` → `05c`, `07` → `07b`, `08` → `08b`, `09` → `09b`) and three files that are not
part of this release (`03`, `04`, `06`). One phrase that named a reviewer's role has been
removed and the heading levels are lowered; the text is otherwise as sent.

---

### Expert review pack

These CSVs are the option lists and prompts for the LLM mental-health treatment audit. We
need clinical review of each before we treat any results as citable. Open in Excel or Google
Sheets; fill the **REVIEW columns** (the ones in CAPS at the right of each file). Leave the
content columns as-is unless you're correcting them via the SUGGESTED EDIT column.

#### What we need from you

**Prompts**
- `01b_patient_scenarios_v2.csv`, `02b_asks_v2.csv`: the current (v2) prompt pack. Each
  prompt is a scenario (part 1) plus an ask (part 2). Three register arms: clinical (a
  clinician), caregiver (a family member), patient (first person). Is each vignette / question
  clinically realistic and fair? Flag anything artificial; suggest edits. (Files 01/02 without
  the "b" are the superseded old versions; ignore them.)
- `03/04_prompts_contraindication_*.csv`: two things per row -- is the **prompt** realistic,
  and is the **expected_correct_action (the KEY)** clinically correct? The key is what a good
  answer must do (flag the risk, advise against, offer a safe alternative). Correct any key
  that's wrong. `04` is the *buried* version (the contraindication is an incidental detail).

**Drugs**
- `05_drugs_generic.csv`: keep/remove each drug; is the drug_class right? Note any missing
  psychiatric drugs.
- `06_drugs_brands.csv`: for each brand, is it a **real brand of
  that generic** (y/n), and is it the **same standard oral formulation** (y/n -- not an
  extended-release, injectable, patch, or combo)? These were gathered from web sources and
  are unverified; some may be wrong or fabricated. This is the gate on the brand-bias result.

**Other option lists**
- `07_digital_apps.csv`: keep/remove; is the internal regulatory tag right? (The tag is
  hidden from the models; we only use it to group results.) Note missing apps.
- `08_brain_stimulation.csv`: keep/remove; note missing procedures. **Lobotomy** was added
  intentionally as a floor item (something that should never be recommended) -- confirm.
- `09_talk_therapies.csv`: keep/remove; note missing therapies. The `alternative` family
  (conversion therapy, recovered-memory therapy, NLP, etc.) is intentionally included as
  fringe/floor items -- confirm these should be present and are correctly flagged.

#### Notes
- Internal tags (drug_class, regulatory, family) are **never shown to the models** -- they
  only group results for analysis. Review them for accuracy, but they don't bias the models.
- Floor items (lobotomy, conversion therapy, hard drugs as "treatment") are deliberate: they
  should rate at the bottom, and a model rating them highly is a finding.
- Add rows for anything important that's missing.
