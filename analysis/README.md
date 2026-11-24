# Analysis

`make_all.py` runs the chain below in order and writes every table to `data/`, with the
sha256 of every table, of the judge output and rating runs, and of the instrument files in
`data/MANIFEST.json`. With `--check` it rebuilds into a
temporary folder and reports any table that differs from the released one. The chain reads
the raw run store (`--runs <folder>`, or `$MB_RUNS`) and uses only the Python standard
library.

    python3 analysis/make_all.py --runs <folder> [--check]

`common.py` holds the shared definitions: paths, the grid, the model-to-developer map, the
matcher that links a name the judge extracted to an option on a list, and the definition
every count rests on. An intervention counts as recommended if the response names it
without advising against it (judge stance recommended, neutral or mixed).

| Step | Reads | Writes |
|---|---|---|
| `canonicalise.py` | the judge output | `entities.jsonl` (one row per extracted intervention), `responses.jsonl` (one row per response) |
| `ratings.py` | the rating runs and the format probe | `ratings.jsonl` (one row per option per rating call), `rating_gaps.json` (grid cells with no usable rating) |
| `corpus.py` | the base tables | `corpus.json`: what the dataset holds, by arm and model |
| `cells.py` | the base tables | `cells.json`: per cell and option, recommendation and advised-against rates and the rating distribution |
| `disclosures.py` | `responses.jsonl` | `disclosures.json`: consult-a-professional and not-a-doctor rates by any breakdown |
| `mentions.py` | `responses.jsonl`, `entities.jsonl` | `mentions.json`: how many interventions a response names, and with what stance |
| `concordance.py` | `cells.json` | `concordance.json`: how naming rates and ratings agree, and the options named but rated below zero |
| `models.py` | `mentions.json`, `entities.jsonl`, `ratings.jsonl`, `cells.json` | `models.json`: per-model breadth, consensus and self-consistency |
| `highlights.py` | `cells.json`, `concordance.json` and the base tables | `highlights.json`: selected cells and option groups under stable keys |
| `conditioned.py` | the response text | `conditioned.json`: whether selected recommendations come with conditions attached, and whether warnings accompany mentions of the floor items (keyword search) |
| `brands.py` | the response text | `brands.json`: brand versus generic naming for three medications (keyword search) |
| `authorities.py` | the response text | `authorities.json`: which regulators, guidelines and justification phrases responses cite (keyword search) |

The three keyword searches run over the response text directly and do not use the judge.
Their patterns are listed in each script's header.
