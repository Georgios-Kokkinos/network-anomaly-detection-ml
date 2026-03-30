# Version 3 - Data Folder Instructions (UNSW-NB15)

Put your UNSW files in this folder before running `python -m src.main`.

Preferred minimum files:
- `Data.csv`
- `Label.csv`

Optional files (can coexist):
- `CICFlowMeter_out.csv`
- `Readme.txt`

Loader behavior:
- If `Data.csv` + `Label.csv` exist, they are used directly.
- Otherwise, the loader scans CSVs and uses files that already contain a label-like column (`Label`, `attack_cat`, `class`, etc.).
