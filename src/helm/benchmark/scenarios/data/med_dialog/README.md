# MedDialog test splits (mirrored)

BioBART test splits for `med_dialog`, mirrored from Codalab bundle
`0x82f0c47f6d3e4462ae9ef8ea39eebe64` (see `med_dialog_scenario.py`).

These files are used from the git checkout for CI and local development. PyPI wheels
also ship them via `MANIFEST.in` (`*.json` under `src/helm/benchmark/`). If bundled
copies are missing, the scenario falls back to GitHub raw:

`https://raw.githubusercontent.com/PacificAI/medhelm/<commit>/src/helm/benchmark/scenarios/data/med_dialog/`

When updating mirrored files, bump `MED_DIALOG_SOURCE_DATA_GIT_HASH` in
`med_dialog_scenario.py` to the commit that contains the new files.
