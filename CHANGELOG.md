# Changelog

The instrument and the releases of results are versioned separately. An instrument version
follows the policy in the README: a major version breaks comparability with earlier runs, a
minor version adds vignettes, options, languages, brands or organizations, and a patch fixes
errors without changing what is measured. A release of results is a set of runs on one
instrument version.

## Release 1.0.0

Instrument 1.0.0. Data in the Hugging Face dataset
[mindbench-ai/intervention-audit](https://huggingface.co/datasets/mindbench-ai/intervention-audit),
revision `v1.0.0`, described in arXiv:2611.12814.

- Generation arm: 41 models, 855 responses each (35,055 in total), collected 2026-08-21 to
  2026-08-22.
- Rating arm: format probe 2026-08-22 to 2026-08-23; 29 qualified models, 684 rating calls
  each (19,836 in total), collected 2026-08-23 to 2026-08-24.
- Judge: GPT-5.6 Luna with the frozen prompt in `run/judge_prompt.py`, chosen on
  2026-08-26 and run over the full generation corpus 2026-08-26 to 2026-08-27. The
  comparison with human coders on the frozen test split was run once, on 2026-09-08.

## Instrument 1.0.0, frozen 2026-08-21

- 19 vignettes in US English (17 conditions and 2 controls), each in three footings:
  clinician, caregiver and patient.
- Five asks: open, medication, psychotherapy, procedures and digital.
- Four option lists: 139 medications, 70 psychotherapies, 35 procedures and 102 digital
  tools. Each list has a null option, and the psychotherapy and procedure lists carry
  floor items (8 and 6).
- Built from the expert-review sheets in `instrument/expert_review/`.
