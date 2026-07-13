---
title: Downloading Raw Results
---
# Downloading Raw Results

MedHELM leaderboard results are stored in Google Cloud Storage (GCS) in the public `crfm-helm-public` bucket. Use the `gcloud storage` CLI ([documentation](https://cloud.google.com/sdk/gcloud/reference/storage)) to download them.

> **Note:** This page covers **MedHELM** results. Historical Stanford HELM projects (Classic, VHELM, AHELM, Image2Struct) may still exist under other paths in the same bucket but are not part of this repository.

## Setup

1. Install `gcloud` following [Google's instructions](https://cloud.google.com/sdk/docs/install). Login is optional for the public bucket.
2. Create a local directory:
```sh
export LOCAL_BENCHMARK_OUTPUT_PATH=./benchmark_output
mkdir -p $LOCAL_BENCHMARK_OUTPUT_PATH
```
3. Set the MedHELM GCS path:
```sh
export GCS_BENCHMARK_OUTPUT_PATH=gs://crfm-helm-public/medhelm/benchmark_output
```

## MedHELM path

- **MedHELM:** `gs://crfm-helm-public/medhelm/benchmark_output`

## Download the full MedHELM tree

Warning: the full tree can be large. Check size before downloading:

```sh
gcloud storage du -sh $GCS_BENCHMARK_OUTPUT_PATH
gcloud storage rsync -r $GCS_BENCHMARK_OUTPUT_PATH $LOCAL_BENCHMARK_OUTPUT_PATH
```

## Download a specific release

1. Set the release version (see [leaderboard.medhelm.org](https://leaderboard.medhelm.org/) for available versions):
```sh
export RELEASE_VERSION=v2.0.0
mkdir -p $LOCAL_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION
gcloud storage rsync -r \
  $GCS_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION \
  $LOCAL_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION
```
2. Inspect `$LOCAL_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION/summary.json` for suite names, then download individual suites if needed (below).

## Download a specific suite

```sh
export SUITE_VERSION=<suite_name_from_summary.json>
mkdir -p $LOCAL_BENCHMARK_OUTPUT_PATH/runs/$SUITE_VERSION
gcloud storage rsync -r \
  $GCS_BENCHMARK_OUTPUT_PATH/runs/$SUITE_VERSION \
  $LOCAL_BENCHMARK_OUTPUT_PATH/runs/$SUITE_VERSION
```

## Troubleshooting

On older `gcloud` versions, `du` or `rsync` subcommands may be missing. Upgrade `gcloud`, or use `gsutil du` / `gsutil rsync` instead ([gsutil docs](https://cloud.google.com/storage/docs/gsutil)).

## GCS browser

Browse files in the [GCS console](https://console.cloud.google.com/storage/browser/crfm-helm-public/medhelm). This requires a Google account and acceptance of GCP Terms of Service.

For a smaller download when you only need aggregated metrics, see [LEADERBOARD_EXPORT.md](https://github.com/PacificAI/medhelm/blob/main/LEADERBOARD_EXPORT.md).
