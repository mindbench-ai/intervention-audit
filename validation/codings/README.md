# Human codings

The human codings used to validate the judge's stance extraction. Eight coders, anonymous
codes R1–R8. Two calibration pilots (R1–R3, ten responses each) and a main wave of 300
responses, each coded by two coders, 75 per coder.

## Files

- `R{i}_pilot1.json`, `R{i}_pilot2.json` — the two calibration pilots: three coders (R1-R3) on the
  same ten responses each, coded under the v1 guide (pilot 1) and the revised guide (pilot 2).
- `R{i}_round1.json` — round 1: the coder's 25 development-split responses, coded under
  the v1 coder guide.
- `R{i}_round2.json` — round 2: the same 25 development responses re-reviewed under the
  amended guide (slot `w1`) plus 50 frozen-test responses (slot `w2`).
- `codings_all.jsonl` — every answer from every file, one line per (coder, round,
  response).
- `manifest.json` — sha256 of every as-run source export and every file here, plus the
  assignment file the packets were built from.

## Schema (per answer)

| field | meaning |
|---|---|
| `rid` | response id: `model|condition|footing|ask|sample` |
| `split` | `dev` (round-1 responses; open for instrument revision) or `test` (frozen) |
| `slot` | `w1` = development response, `w2` = frozen-test response |
| `seconds` | time on the response in the coding interface (`null` for prefilled re-reviews the coder did not reopen) |
| `stances` | judge-extracted entity surface → coder's answer: `recommended`, `neutral`, `advised_against`, `mixed`, `not_applicable` (= "not a treatment or resource"), or `""` (unanswered) |
| `missed` | free text: interventions the coder found that the judge did not extract |
| `not_a_doctor` | checkbox: response says it is not a doctor / not qualified |
| `consult_professional` | checkbox: response tells the reader to consult a professional |

Entity surfaces are exactly the strings the judge extracted and the packet highlighted;
a response with no judge entities has an empty `stances` object and is still a valid
answer for the checkboxes and `missed`.

## Provenance and transforms

Produced by `validation/ingest_codings.py` from the as-run exports, which are kept unedited
and are not released because they identify the coders. Pilot exports were tagged with coder
initials and are mapped to the R-codes through a file that is not released. Three coders (R6–R8)
used a clipboard export because mail filters blocked file attachments; R7 and R8 pasted
it into Word. The ingest applies only lossless repairs, each logged in the file's
`repairs` list: docx→text extraction; re-escaping of quote characters inside entity
names that the clipboard export wrote unescaped; restoring typographic quotes that a
text editor straightened; merging a duplicate key that the interface exported with
rendered HTML instead of the judge's markdown surface; and reversing the URL defanging
(a space after `https://`, `http://`, `www.`) that the coder-facing packets carried
for mail deliverability.

Every answer was validated against `../assignments.json` (rid assigned to that coder
and wave; 25 + 50 per round-2 file) and against the entity list shipped in that
coder's packet HTML.

## Anonymity

Coders appear only as R1–R8. The mapping from codes to individuals is held by the study
lead outside the repository and is not part of any release.
