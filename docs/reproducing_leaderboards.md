---
title: Reproducing Leaderboards
---
# Reproducing Leaderboards

Use MedHELM to rerun evaluation runs and reproduce the public [MedHELM leaderboard](https://leaderboard.medhelm.org/).

Configuration files live in the repository:

- **Run entries:** `src/helm/benchmark/presentation/run_entries_medhelm_*.conf`
- **Schema:** `src/helm/benchmark/static/schema_medhelm.yaml`

## General procedure

```bash
# Pick any suite name
export SUITE_NAME=my_suite

# Replace with your model(s)
export MODELS_TO_RUN=openai/gpt-4o-2024-05-13

# MedHELM public benchmarks (see access levels below)
export RUN_ENTRIES_CONF_PATH=src/helm/benchmark/presentation/run_entries_medhelm_public.conf
export SCHEMA_PATH=src/helm/benchmark/static/schema_medhelm.yaml
export NUM_TRAIN_TRIALS=1
export MAX_EVAL_INSTANCES=1000
export PRIORITY=2

medhelm-run \
  --conf-paths "$RUN_ENTRIES_CONF_PATH" \
  --num-train-trials "$NUM_TRAIN_TRIALS" \
  --max-eval-instances "$MAX_EVAL_INSTANCES" \
  --priority "$PRIORITY" \
  --suite "$SUITE_NAME" \
  --models-to-run "$MODELS_TO_RUN"

helm-summarize --schema "$SCHEMA_PATH" --suite "$SUITE_NAME"

helm-server --suite "$SUITE_NAME"
```

Then open the local frontend (typically http://localhost:8000).

## Benchmark access levels

MedHELM benchmarks are grouped by data access. See [Benchmark Access Levels](/medhelm#benchmark-access-levels) for details and example sources.

### Public benchmarks

Fully open and freely available.

```bash
export RUN_ENTRIES_CONF_PATH=src/helm/benchmark/presentation/run_entries_medhelm_public.conf
export SCHEMA_PATH=src/helm/benchmark/static/schema_medhelm.yaml
export NUM_TRAIN_TRIALS=1
export MAX_EVAL_INSTANCES=1000
export PRIORITY=2
```

### Gated benchmarks

Publicly available but require credentials or approval (e.g. PhysioNet, Hugging Face gated datasets, Google Drive downloads).

```bash
export RUN_ENTRIES_CONF_PATH=src/helm/benchmark/presentation/run_entries_medhelm_gated.conf
export SCHEMA_PATH=src/helm/benchmark/static/schema_medhelm.yaml
export NUM_TRAIN_TRIALS=1
export MAX_EVAL_INSTANCES=1000
export PRIORITY=2
```

Install gated dependencies: `pip install "medhelm[gated]"`.

### Private benchmarks

Accessible only to specific organizations. Use the appropriate private config for your org, for example:

```bash
export RUN_ENTRIES_CONF_PATH=src/helm/benchmark/presentation/run_entries_medhelm_private_stanford.conf
export SCHEMA_PATH=src/helm/benchmark/static/schema_medhelm.yaml
export NUM_TRAIN_TRIALS=1
export MAX_EVAL_INSTANCES=1000
export PRIORITY=2
```

Private configs expect local data paths to be configured for your environment.

### DSPy evaluation runs

For MedHELM runs with DSPy-optimized agents:

```bash
export RUN_ENTRIES_CONF_PATH=src/helm/benchmark/presentation/run_entries_medhelm_dspy.conf
```

Install the DSPy extra: `pip install "medhelm[dspy]"`.

## Downloading existing leaderboard results

To compare against published results without rerunning everything, download from Google Cloud Storage. See [Downloading Raw Results](/downloading_raw_results) and [MedHELM — Viewing leaderboard results](/medhelm#viewing-and-reproducing-leaderboard-results).
