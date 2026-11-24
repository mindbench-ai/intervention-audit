# Data

The derived tables of the current release, written by the analysis chain, and
`MANIFEST.json`, which records their hashes. [`analysis/README.md`](../analysis/README.md)
says which script writes each table and what it holds.

| File | Contents |
|---|---|
| `responses.jsonl` | One row per open-ended response, with its cell, the counts of named interventions, and the disclosure flags |
| `entities.jsonl` | Every intervention the judge extracted, with its stance and the option it maps to (downloaded by `fetch.py`) |
| `ratings.jsonl` | One row per option per rating call (downloaded by `fetch.py`) |
| `rating_gaps.json` | Rating cells with no usable call, with every attempt |
| `corpus.json`, `cells.json`, `disclosures.json`, `mentions.json`, `concordance.json`, `models.json`, `highlights.json` | Summaries by cell, footing and model |
| `conditioned.json`, `brands.json`, `authorities.json` | The keyword analyses of the response text |
| `fetch.py` | Downloads a release from Hugging Face |

`entities.jsonl` and `ratings.jsonl` are too large for GitHub. `python3 data/fetch.py`
downloads them from the Hugging Face dataset
[mindbench-ai/intervention-audit](https://huggingface.co/datasets/mindbench-ai/intervention-audit),
and with `--runs <folder>` also rebuilds the raw run store that the analysis chain reads.
