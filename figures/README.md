# Figures

Each script reads the tables in `data/` (`disclosures_by_model.py` also reads the model
roster in `config/models.json`) and writes a PDF and a PNG to `out/` under its own name. `figstyle.py` holds the shared style and loads the fonts in `fonts/`, and `cellutil.py`
holds the labels and loaders shared by the per-cell figures.

    python3 figures/scripts/<script>.py

| Script | Shows | Reads |
|---|---|---|
| `overview_map.py` | Every cell's options ranked by recommendation rate, colored by mean rating | `cells.json` |
| `concordance.py` | Recommendation rate against mean rating for every option, on each list's own ask and on the open ask (`--numbered` writes `concordance_numbered`) | `cells.json` |
| `disclosures.py` | Consult-a-professional and not-a-doctor rates by footing and ask | `disclosures.json` |
| `disclosures_by_model.py` | The same rates for each model, by footing | `disclosures.json`, `config/models.json` |
| `model_breadth.py` | Each model's recommended interventions per response against its share of recommendations the rating panel scored below zero | `models.json` |
| `self_consistency.py` | For each rated model, the share of its own recommendations that it rated below zero | `models.json` |
| `lisdexamfetamine_bed.py` | Lisdexamfetamine for binge-eating disorder, model by model and by footing | `highlights.json` |
| `catalogue_cells.py` | One page per cell (76), in `out/catalogue/` | `cells.json` |
