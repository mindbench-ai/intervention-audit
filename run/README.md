# Runners

These scripts send the instrument to the models and record what comes back. Each one
appends JSONL rows to a folder of the run store (the `--out` folder). Every row records the
instrument version, the model's served name and the telemetry the provider returned. The
generation, rating and judge runners resume where they stopped. The options for each
script are in its header.

## Steps

1. **Models.** Add each model to `config/models.json` (provider, model id, base URL, API
   key variable, developer `family`, tier) and put the API keys in `.env` at the repository root.
2. **Generation arm.** `run_lanes.py` runs `gen_run.py` against the live APIs, one lane
   per API account; `batch_gen_run.py` uses the batch APIs of Anthropic, OpenAI and Google.
   Both write the same rows. `export_run.py` then writes the usable responses of a run as
   one file per model under `export/sidecar/`, which the judge and the analysis read.
3. **Rating arm.** `rate_qualify.py` runs the format probe (24 calls per model) and
   `validation/qualify_report.py` scores it against the pre-specified gate. The qualified
   models then run the full grid through `rate_run.py` (live) or `batch_rate_run.py`
   (batch). `rating_instrument.py` holds the rating prompt, the API call and the parser
   that all three share.
4. **Judge.** `batch_judge_run.py` sends every exported response, after the frozen prompt
   in `judge_prompt.py`, to the judge model through the OpenAI Batch API. The prompt's
   sha256 is stamped on every row, and importing `judge_prompt.py` fails if its text has
   changed.
5. **Analysis.** `python3 analysis/make_all.py --runs <folder>` (see `analysis/`).

`batch_backends/` holds the clients for the three batch APIs (standard library only).

## The run store of release 1.0.0

| Folder | Contents |
|---|---|
| `runs/gen_v1/<lane>/` | Generation arm, 35 models, one folder per lane (`anthropic-batch`, `google-batch`, `openai-batch`, `deepseek`, `mistral`, `moonshot`, `qwen`, `xai`), 2026-08-21 to 2026-08-22 |
| `runs/legacy_v1/` | Generation arm, the six older models (GPT-3.5 Turbo, the August 2025 snapshots of GPT-5 and GPT-5 mini, and Gemini 2.5 Pro, Flash and Flash-Lite), run in one launch, 2026-08-21 to 2026-08-22 |
| `runs/gen_v1/export/`, `runs/legacy_v1/export/` | The usable responses of each generation run, written by `export_run.py` |
| `runs/qualify/probe.jsonl` | The rating-arm format probe, 2026-08-22 to 2026-08-23 |
| `runs/rating_v1/<lane>/ratings.jsonl` | Rating arm, 29 models, 2026-08-23 to 2026-08-24 |
| `runs/judge_v1/luna/judgments.jsonl` | The judge pass (GPT-5.6 Luna), 2026-08-26 to 2026-08-27 |

In raw rows the footing is the field `register`, the ask is `category` (generation and
judge rows), and the option list is `scope` (rating rows); the analysis renames them
`footing`, `ask` and `list`.

The store that `data/fetch.py` rebuilds from Hugging Face holds the same rows, with all 41
models' responses under `runs/gen_v1/export/sidecar/` and the rating rows of every lane in
`runs/rating_v1/all/`.

## Run folders

The judge and the analysis read the runs of release 1.0.0 from these folders by name:
`gen_v1` and `legacy_v1` in `analysis/common.py`, `run/batch_judge_run.py` and
`validation/build_pilot_packet.py`; `judge_v1/luna` in `analysis/common.py`,
`analysis/make_all.py` and `validation/judge_comparison.py`; `rating_v1` and `qualify` in
`analysis/ratings.py`, `analysis/make_all.py` and `run/rate_run.py`. A new release sets
its own folder names there.
