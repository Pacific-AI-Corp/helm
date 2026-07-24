---
title: Efficient MedHELM
---
# Efficient MedHELM

This tutorial shows how to add your model to the [MedHELM leaderboard](https://leaderboard.medhelm.org/#/leaderboard) at a fraction of the cost of a full run, using the technique from IBM Research described in [Efficient Benchmarking (of Language Models)](https://arxiv.org/pdf/2308.11696.pdf) (Perlitz et al., 2023).

The rank-location tradeoffs in the table below were measured on the original HELM Classic benchmark suite. MedHELM uses the same `--max-eval-instances` subsampling approach; treat the rank CI numbers as approximate guidance rather than MedHELM-specific calibration.

## Prerequisites

- MedHELM installed ([installation guide](/installation))
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`) for downloading public leaderboard results
- API credentials or local deployments configured for your model ([adding new models](/adding_new_models))

## Download MedHELM leaderboard results

To compare your model against models on the official leaderboard, download the public MedHELM release tree from Google Cloud Storage:

```bash
export OUTPUT_PATH="./benchmark_output"
export GCS_BENCHMARK_OUTPUT_PATH="gs://crfm-helm-public/medhelm/benchmark_output"
export RELEASE=v2.0.0

mkdir -p "$OUTPUT_PATH"

# Optional: check download size first
# gcloud storage du -sr "$GCS_BENCHMARK_OUTPUT_PATH"

gcloud storage rsync -r "$GCS_BENCHMARK_OUTPUT_PATH" "$OUTPUT_PATH"
```

After the download, confirm that release metadata exists, for example `benchmark_output/releases/v2.0.0/summary.json`. Pick the `RELEASE` value from the version selector on [leaderboard.medhelm.org](https://leaderboard.medhelm.org/).

> **Note:** MedHELM does not publish a lightweight `run_stats.zip` archive. Leaderboard results are distributed via the GCS path above. See [Viewing and Reproducing Leaderboard Results](/medhelm#viewing-and-reproducing-leaderboard-results) for more detail.

### Optional: smaller download

If you only need aggregated metrics (not per-instance request/response files), you can sync from an existing local copy while excluding large artifacts. See [LEADERBOARD_EXPORT.md](https://github.com/PacificAI/medhelm/blob/main/LEADERBOARD_EXPORT.md) in the repository.

## Run efficient MedHELM

According to [Efficient Benchmarking (of Language Models)](https://arxiv.org/pdf/2308.11696.pdf), running with a fraction of evaluation instances per scenario can still yield a reliable estimate of a full run. The authors report the following tradeoffs[^1]:

| Examples Per Scenario | CI 95% of Rank Location | Compute saved |
| :-------------------: | :---------------------: | :-----------: |
|          10           |           ±5            |     X400      |
|          20           |           ±4            |     X200      |
|          50           |           ±3            |      X80      |
|          200          |           ±2            |      X20      |
|         1000          |           ±1            |      X4       |
|          All          |           ±1            |      X1       |

Choose your tradeoff, then set your model and evaluation depth:

```bash
export EXAMPLES_PER_SCENARIO=10
export MODEL=openai/gpt2
export MODEL_DEPLOYMENT=huggingface/gpt2
export SCHEMA_PATH=src/helm/benchmark/static/schema_medhelm.yaml
```

`run_entries_medhelm_public.conf` lists the models and Stanford Healthcare deployments used on the public leaderboard. To evaluate **your** model on the same scenarios, copy the config and replace `model=` / `model_deployment=` in each entry:

```bash
cp src/helm/benchmark/presentation/run_entries_medhelm_public.conf run_entries_my_model.conf
# Edit run_entries_my_model.conf: set your MODEL and MODEL_DEPLOYMENT on every entry.
```

For a quick smoke test on a few public scenarios before running the full public config:

```bash
medhelm-run \
  --run-entries "pubmed_qa:model=$MODEL,model_deployment=$MODEL_DEPLOYMENT" \
                "medcalc_bench:model=$MODEL,model_deployment=$MODEL_DEPLOYMENT" \
  --suite "$RELEASE" \
  --output-path "$OUTPUT_PATH" \
  --max-eval-instances "$EXAMPLES_PER_SCENARIO" \
  --num-train-trials 1 \
  --cache-instances \
  --skip-completed-runs
```

To run the full public benchmark suite with subsampled instances:

```bash
medhelm-run \
  --conf-paths run_entries_my_model.conf \
  --suite "$RELEASE" \
  --output-path "$OUTPUT_PATH" \
  --max-eval-instances "$EXAMPLES_PER_SCENARIO" \
  --num-train-trials 1 \
  --cache-instances \
  --skip-completed-runs \
  --priority 2
```

Use the same `--suite` value as the release you downloaded (for example `v2.0.0`) so your runs appear alongside existing leaderboard results. `--skip-completed-runs` avoids re-running models that are already present under `benchmark_output/runs/$RELEASE/`.

The first run may take a while because datasets are downloaded and cached regardless of how many instances you evaluate.

## Summarize and serve your results

Aggregate results with the MedHELM schema:

```bash
helm-summarize --schema "$SCHEMA_PATH" --suite "$RELEASE" -o "$OUTPUT_PATH"
```

Launch the local leaderboard UI with the official release:

```bash
helm-server --release "$RELEASE" --output-path "$OUTPUT_PATH"
```

Open `http://localhost:8000` to see your model next to the downloaded leaderboard models.

For a full reproduction (all instances, gated/private benchmarks), see [Reproducing Leaderboards](/reproducing_leaderboards#medhelm).

## References

Perlitz, Y., Bandel, E., Gera, A., Arviv, O., Ein-Dor, L., Shnarch, E., Slonim, N., Shmueli-Scheuer, M. and Choshen, L., 2023. Efficient Benchmarking (of Language Models). arXiv preprint arXiv:2308.11696.

[^1]: The quantities above are the CI 95% of rank location and are conservative estimates. In our experiments on HELM Classic, deviations were typically within ±2 ranks for the subsampling levels above.
