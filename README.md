# MindBench Intervention Audit

The MindBench Intervention Audit tracks which mental health treatments and interventions
large language models recommend, and how those recommendations change depending on how a
request is framed and who is asking. It is one of the audits on the
[MindBench](https://mindbench.ai) platform, and its results are published at
[mindbench.ai/audits/interventions](https://mindbench.ai/audits/interventions). The audit and its
first release are described in [arXiv:2611.12814](https://arxiv.org/abs/2611.12814), and the full
corpus of each release is in the Hugging Face dataset
[mindbench-ai/intervention-audit](https://huggingface.co/datasets/mindbench-ai/intervention-audit).

## How the audit works

**Instrument.** A versioned set of files in `instrument/` defines the content of the audit.
It contains the case vignettes, each of which is written in several footings, and the
questions (asks) that follow a vignette, which are either open or about a single
intervention type. It also contains an option list for each intervention type, with the
brand names and aliases used to match different names for the same intervention. The
instrument is frozen with a hash manifest before each run, and every row that a runner
writes records the instrument version it used.

**Open-ended generation arm.** Each vignette and ask is sent to each model several times as
a single user message with no system prompt, and the runners record each response together
with the telemetry returned by the provider.

**Judge.** An LLM judge with a frozen prompt extracts every intervention that a response
names, together with the model's stance toward it, and the extracted names are then matched
to the option lists. The judge is validated against human coders, and the coder guide,
coding packets, codings, and agreement scripts are in `validation/`.

**Structured rating arm.** Models rate every option on a list for the same vignettes, with
the order of the list shuffled for each call. Each model must pass a format check before
its full rating run.

**Analysis.** A single deterministic chain of scripts turns the raw run rows into tables by
vignette and intervention type and records the hash of every input and output. The figures
are built from these tables and the model roster.

## Terminology

Some files and fields use different names for the paper's terms.

| In the paper | In files and code |
|---|---|
| vignette | `instrument/dataset/situations.json`, keyed by condition; "presentation" in code comments, "scenario" in the review sheet |
| footing (clinician, caregiver, patient) | `clinical`, `caregiver`, `patient`; the field `register` in raw run rows and `footing` in the tables in `data/`; "arm" in the vignette review sheet and the style guide |
| ask | the field `category` in raw generation and judge rows, `ask` in the tables in `data/` |
| option list (medications, psychotherapies, procedures, digital tools) | the files in `instrument/lexicon/`; the field `scope` in raw rating rows and `list` in the tables in `data/`, with the values `pharmacological`, `psychotherapy`, `neuromodulation` and `digital` |
| cell | one vignette and one option list (19 × 4 = 76 cells) in the analysis and figures; in the runners, `data/rating_gaps.json` and the dataset card, one slot of the run grid (one model, vignette, footing, ask or list, and sample) |
| floor item | a discredited or historical practice carried on a list on purpose |
| generation arm, rating arm | open-ended responses, and ratings of every option on a list. In the code, `arm` can also name a run folder (`gen_v1`, `legacy_v1`), as in the `arm` field of judge rows |
| recommended | named without being advised against: judge stance `recommended`, `neutral` or `mixed` (`analysis/common.py`) |

## Repository layout

| Folder | Contents |
|---|---|
| `instrument/` | Vignettes, asks, option lists, the expert-review source sheets they are built from, and the version manifest |
| `config/` | The model roster and API settings |
| `run/` | Runners for the generation arm, the rating arm, and the judge, and the frozen judge prompt |
| `validation/` | Human validation of the judge |
| `analysis/` | The analysis chain, from raw run rows to the tables in `data/` |
| `figures/` | Figure scripts (`scripts/`), which read the tables in `data/`, the rendered figures (`out/`), and the fonts they use (`fonts/`) |
| `data/` | Derived tables for the current release |

The `instrument/`, `run/`, `validation/`, `analysis/` and `figures/` folders each have a README
that goes through their files in order, and `CHANGELOG.md` records
each instrument version and release.

## Running the audit

The code requires Python 3.12 or later. To install the required packages, run

    pip install -r requirements.txt

To audit new models, add them to `config/models.json` and put the provider API keys in a
`.env` file at the repository root. The runners read `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GOOGLE_API_KEY` (or `GEMINI_API_KEY`), `MISTRAL_API_KEY`, `XAI_API_KEY`,
`DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`, and `DASHSCOPE_API_KEY`.

- Generation: `run/run_lanes.py` for live API calls or `run/batch_gen_run.py` for providers
  with batch endpoints, followed by `run/export_run.py`
- Rating: `run/rate_qualify.py` for the format check, followed by `run/rate_run.py` or
  `run/batch_rate_run.py`
- Judge: `run/batch_judge_run.py`

The options for each script are documented in its header. The runners read the vignettes,
asks, and option lists from `instrument/`, but the footings are also named in
`run/gen_run.py` and `run/rate_run.py`, and the analysis lists its footings, asks, and option
lists in `analysis/common.py`, so adding a footing or an intervention type requires changes
to these files as well as to the instrument.

## Data

`data/` holds the derived tables for the current release, and `data/MANIFEST.json` records
the hash of every input and output.

- `responses.jsonl`: one row per open-ended response, with its cell, the counts of named
  interventions, and the disclosure flags
- `entities.jsonl`: every intervention the judge extracted, with its stance and the option it
  maps to
- `ratings.jsonl`: one row per option per rating call
- the remaining `.json` files: summaries by cell, footing, and model, and the keyword analyses

`entities.jsonl` and `ratings.jsonl` are too large for GitHub and are hosted with the full
corpus in the Hugging Face dataset
[mindbench-ai/intervention-audit](https://huggingface.co/datasets/mindbench-ai/intervention-audit),
which holds every open-ended prompt and response, every judge output, and every rating call for
a release together with its provenance, under CC BY 4.0. The rating prompts are not stored, but
`run/rating_instrument.py` rebuilds each one exactly, since its option order is seeded by the cell.
`python3 data/fetch.py` downloads the two tables into `data/`.

## Reproducing a release

    python3 instrument/check_pack.py                      # instrument matches its manifest
    python3 data/fetch.py --runs <folder>                 # download a release and its raw runs
    python3 analysis/make_all.py --runs <folder> --check  # rebuild matches data/
    python3 figures/scripts/<figure>.py                   # redraw a figure from data/

`data/fetch.py` restores the derived tables from Hugging Face and, with `--runs`, rebuilds the
raw run store that the analysis chain reads. `data/fetch.py`, `instrument/check_pack.py` and the
analysis chain use only the Python standard library; the runners and the figures need the
packages in `requirements.txt`.

## Versioning

The instrument follows semantic versioning, in which major versions break comparability with
earlier runs, minor versions add vignettes, options, languages, brands, or organizations, and
patch versions fix errors without changing what is measured. The audit is designed to be
repeated as models are released and retired, and interventions that models name but that are
missing from the option lists stay in `data/entities.jsonl`, with no option, as candidates for
the next version of the lists.

## Use of generative AI

The authors defined the audit's data models and logic in specifications and pseudocode,
which were fleshed out into code and debugged in Claude Code, and the authors reviewed and
tested the resulting code. That code includes the runners that prompted the audited models
and the judge, the scripts that built the human coding packets, the analysis chain, and the
figure scripts. The first versions of the scripts were written with Claude Fable 5 and
Claude Fable 5.1, with Claude Opus 5 and Claude Sonnet 5 as subagents, and the later work on
this repository used Claude Opus 5.5 and Claude Opus 5. Claude Code subagents also searched
for missing options when the option lists were refined, and Claude provided copyediting
support for the initial drafts of the vignettes, the coder guides, the review instructions
and the judge prompts. Claude Opus 5.5 drafted this README, the folder READMEs, the
changelog and the comments throughout the code, and wrote the scripts that build the Hugging
Face dataset. The authors read and edited every README, reviewed the comments, and reviewed
and edited all other written content for accuracy and fidelity to the study specifications.

## Citation

The audit and its first release are described in *Curbside Chatbots: Auditing the Mental
Health Interventions Large Language Models Recommend to Clinicians, Caregivers, and
Patients* ([arXiv:2611.12814](https://arxiv.org/abs/2611.12814)). Please cite this paper if
you use the audit, its instrument, or its data. Citation details are in `CITATION.cff`.

## License

Code is released under the MIT License (`LICENSE`). The instrument, data, validation
materials, and figures are released under CC BY 4.0 (`LICENSE-DATA.md`), and the fonts in
`figures/fonts/` are distributed under the SIL Open Font License (`figures/fonts/OFL.txt`).

## Contact

Please use the contact form at [mindbench.ai/contact](https://mindbench.ai/contact).
