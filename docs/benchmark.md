---
title: Advanced Benchmarking Guide
---
# Advanced Benchmarking Guide

This guide covers advanced `medhelm-run` options for MedHELM benchmarks. For a first run, see [Quick Start](/quick_start) or [MedHELM](/medhelm).

## Dry runs

Use dry runs to verify configuration and scenario loading without sending requests to a model.

```bash
# Load the config and build instances only (no model requests)
medhelm-run \
  --conf-paths src/helm/benchmark/presentation/run_entries_medhelm_public.conf \
  --max-eval-instances 10 \
  --suite v1 \
  --skip-instances

# Build instances and requests, but do not call the model
medhelm-run \
  --conf-paths src/helm/benchmark/presentation/run_entries_medhelm_public.conf \
  --max-eval-instances 10 \
  --suite v1 \
  --dry-run
```

For a single scenario:

```bash
medhelm-run --run-entries med_qa:model=openai/gpt2 --suite v1 --max-eval-instances 10 --dry-run
```

## Estimating token usage

Append `--dry-run` to estimate token usage without making API calls:

```bash
medhelm-run -r <run_spec_name>:model=<model> --suite $SUITE --max-eval-instances 10 --dry-run
```

Check the output under `benchmark_output/runs/$SUITE`. The `sum` field estimates total tokens for that run spec.

For OpenAI models, token counts use a GPT-2 tokenizer downloaded and cached on first dry run.

## Private and gated benchmarks

MedHELM benchmarks fall under three access levels: **public**, **gated**, and **private**. See [Benchmark Access Levels](/medhelm#benchmark-access-levels) on the MedHELM page.

- **Public:** `run_entries_medhelm_public.conf`
- **Gated** (e.g. MedQA, MedMCQA via Google Drive): `run_entries_medhelm_gated.conf` — requires `pip install "medhelm[gated]"` and a valid `HF_TOKEN` where applicable
- **Private** (organization-specific): e.g. `run_entries_medhelm_private_stanford.conf` — requires local data paths configured for your organization

## Perspective API (optional)

Some proxy workflows use Google's [Perspective API](https://www.perspectiveapi.com) to score toxicity of completions. To enable it, generate an API key from GCP following the [Get Started](https://developers.perspectiveapi.com/s/docs-get-started) and [Enable the API](https://developers.perspectiveapi.com/s/docs-enable-the-api) guides, then add to `credentials.conf`:

```
perspectiveApiKey: <Generated API key>
```

Install the metrics extra if needed: `pip install "medhelm[metrics]"`.

By default, Perspective API allows 1 query per second. Use [this form](https://developers.perspectiveapi.com/s/request-quota-increase) to request a higher quota.
