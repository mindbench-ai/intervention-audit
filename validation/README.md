# Validation

The human validation of the judge, and the format gate for the rating arm, in the order
they were run. The scripts that build packets or call the judge read the raw run store;
the agreement scripts read only files in this folder.

The packets are self-contained HTML files: GitHub shows their source, so download one and open
it in a browser to see it as the coders did.

The coder guide changed between stages: pilot 2 used a revised version of the pilot 1 guide,
round 1 used the pilot 2 guide with two additions, and round 2 added three amendments, shown
in magenta in the round-2 packets. Each packet carries the guide its coders read.

## Judge selection

| Script | Writes | Purpose |
|---|---|---|
| `build_judge_dev_packet.py` | `packets/demo_packet_judges.html` | Twelve responses annotated by each of four candidate judges with the same prompt, for choosing the judge. GPT-5.6 Luna was chosen on cost and quality on 2026-08-26. The script now calls only Luna, and re-running it overwrites the released packet |
| `build_heldout_test.py` | `test20_luna_r1.jsonl` | Responses never used in judge development, annotated with the judge prompt for human review before the prompt was frozen for the corpus pass (19 responses) |

## Calibration pilots

| Script or file | Purpose |
|---|---|
| `pilot_instructions.md` | The instructions sent to coders with the first pilot packet; the later packets came with the same instructions |
| `build_pilot_packet.py` | The packet code, which the main wave reuses: it builds one self-contained HTML file with the coder guide, the responses and the judge's highlighted entities. `packets/pilot_packet.html` and `packets/pilot2_packet.html` are the packets as sent for the two pilots, with the responses listed in `pilot10_selection.json` (pilot 1) and `pilot2_selection.json`. Pass `--out` when re-running it: the default path replaces the packet as sent. `lexicon/substances.json` adds substance names to the highlights |
| `pilot_agreement.py pilot1\|pilot2` | Agreement among the three pilot coders, against the pre-specified gates |

## Main wave

| Script | Writes | Purpose |
|---|---|---|
| `build_main_wave.py` | `assignments.json`, `packets/main_wave/` | 300 responses, each assigned to two of eight coders. Round 1 is the 100-response development split; round 2 is the 200-response frozen test split, with each coder's round-1 responses carried over for re-review under the amended guide. The sample excludes every response used in judge development and in the pilots |
| `checkpoint_round1.py` | (report) | Human-human agreement on round 1, computed before round 2 was released |
| `ingest_codings.py` | `codings/` | Turns the coders' exports into the released codings. The exports are not released because they identify the coders; see `codings/README.md` |
| `agreement_humans.py` | `agreement_humans.json` | Human-human agreement on the main wave |
| `judge_comparison.py` | `judge_comparison.json` | Judge-human against human-human agreement on the frozen test split, on the pre-specified endpoints. Run once, on 2026-09-08 |

## Rating-arm format gate

| Script | Writes | Purpose |
|---|---|---|
| `qualify_report.py` | `qualification_verdicts.json` | Scores each model's format probe (`run/rate_qualify.py`) against the pre-specified gate: at least 80% of its calls must cover at least 90% of the options. The gate reads format only, never what the ratings say |
