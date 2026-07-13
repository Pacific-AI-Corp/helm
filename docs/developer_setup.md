---
title: Developer Setup
---
# Developer Setup

This guide is for contributors working on the MedHELM repository. For end-user installation, see [Installation](/installation).

## Requirements

- Python **3.10**, **3.11**, or **3.12** (3.12 recommended)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — used by CI and recommended for development

Check your Python version:

```bash
python --version
```

## Set up the environment

From the repository root:

```bash
# Create and activate a virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install the package plus dev tools and CI dependencies (matches GitHub Actions)
uv sync --extra ci
```

The `dev` and `types` dependency groups are included automatically via `[tool.uv] default-groups`.

## Run tests

CI runs unit tests excluding slow integration markers:

```bash
uv run pytest -m "not models and not scenarios" --durations=20
```

Scenario integration tests (network downloads):

```bash
uv sync --extra ci --extra scenarios
uv run pytest -m scenarios
```

Run a specific test file:

```bash
uv run pytest src/helm/benchmark/scenarios/test_medi_qa_scenario.py -vv
```

## Run linter and type-checker

Install pre-commit hooks:

```bash
pre-commit install
```

Run manually:

```bash
./pre-commit.sh
```

Or individually:

```bash
black src scripts
flake8 src scripts
mypy src scripts
```

## Executing commands with local changes

After `uv sync`, CLI entry points use your local checkout:

```bash
medhelm-run --run-entries med_qa:model=openai/gpt2 --suite dev --max-eval-instances 5
helm-summarize --schema src/helm/benchmark/static/schema_medhelm.yaml --suite dev
helm-server --suite dev
```

### Without installing

From the repository root:

```bash
PYTHONPATH=src uv run python -m helm.benchmark.run --help
```

## Checking in code

MedHELM uses pull requests against `main`. Typical workflow:

1. `git checkout main && git pull origin main`
2. Make changes and run tests locally
3. `git checkout -b <your-handle>/<change-id>`
4. `./pre-commit.sh` (or rely on pre-commit hooks on push)
5. `git commit -a` and `git push origin <branch>`
6. Open a PR on [GitHub](https://github.com/PacificAI/medhelm)
