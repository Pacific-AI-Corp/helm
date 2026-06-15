# MedHELM documentation (Jekyll)

**Live documentation:** [medhelm.org](https://medhelm.org)

This folder is the source for the site. It is built with **Jekyll** and deployed to **gh-pages** via a GitHub Action on push to **main** (when `docs/**` or the workflow file changes). Configure GitHub Pages to serve from the **gh-pages** branch (Settings → Pages → Source: Deploy from a branch → Branch: gh-pages, / (root)).

## MedHELM library (quick reference)

Documentation on [medhelm.org](https://medhelm.org) covers the full workflow. For a complete step-by-step guide, see the [repository README](../README.md). Summary:

### Getting started from a git clone (development)

Follow these steps **in order**. Skipping a step is a common cause of `command not found: uv`, `ModuleNotFoundError: helm.benchmark.static_build`, or `bus error` on Apple Silicon.

#### 0. Install tools (once per machine)

**uv** (manages Python and the virtual environment):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # add uv to PATH; restart the terminal or add this to ~/.zshrc
uv --version
```

**Node.js 18+** (only needed to build the web UI for `helm-server`):

```sh
# macOS (Homebrew): brew install node
# Or use fnm/nvm — see https://nodejs.org/
node --version
npm --version
```

You do **not** need to install Python separately: `uv venv --python 3.12` downloads Python 3.12 for you.

#### 1. Clone and enter the repository

```sh
git clone https://github.com/PacificAI/medhelm.git
cd medhelm
```

#### 2. Create and activate a virtual environment

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
```

#### 3. Install MedHELM (editable / development mode)

```sh
uv pip install -e .
```

#### 4. Build the web UI (required before `helm-server`)

The React UI is not shipped inside the git clone; build it once:

```sh
cd helm-frontend
npm install
npm run build -- --outDir '../src/helm/benchmark/static_build' --emptyOutDir
cd ..
```

#### 5. Run a quick benchmark and open the results

```sh
medhelm-run --run-entries "pubmed_qa:model=openai/gpt2,model_deployment=huggingface/gpt2" --suite my_med_test --max-eval-instances 2
helm-summarize --suite my_med_test -o ./benchmark_output
helm-server --suite my_med_test -o ./benchmark_output --port 8000
```

Open **http://localhost:8000** in your browser (not `0.0.0.0`). Press `Ctrl-C` in the terminal to stop the server.

Local Hugging Face models (e.g. `gpt2`) automatically use the best available PyTorch device: **CUDA → Apple MPS → CPU**.

### Getting started from PyPI (no git clone)

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install medhelm
```

The PyPI package includes a pre-built web UI (no Node.js required). Then run the commands in step 5 above.

### Tiers

| Tier | Install | Scenarios |
|------|--------|-----------|
| **Standard** | `uv pip install -e .` (repo) or `pip install medhelm` / `uv pip install medhelm` (PyPI) | PubMedQA, MedCalc-Bench, MedicationQA, MedHallu |
| **Summarization** (Clinical NLP tier) | `pip install "medhelm[summarization]"` | DischargeMe, ACI-Bench, Patient-Edu (install may take 2–3 min; adds bert-score, rouge-score, nltk) |
| **Gated** (licensing tier) | `pip install "medhelm[gated]"` | MedQA, MedMCQA (Google Drive; adds gdown) |

**Full example** (better quality, 10 instances; needs more RAM/VRAM):

```sh
medhelm-run --run-entries "pubmed_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" --suite my_med_test --max-eval-instances 10
helm-summarize --suite my_med_test -o ./benchmark_output
helm-server --suite my_med_test -o ./benchmark_output --port 8000
```

## Local build (Jekyll)

**Ruby >= 3.0 is required** (Ruby 3.x and Ruby 4.x are supported). macOS ships with Ruby 2.6; install a newer Ruby with Homebrew:

```bash
brew install ruby
# Add to your PATH (e.g. in ~/.zshrc): export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
# Then:
cd docs
bundle install
```

### Full site preview (recommended)

Several pages (`models.md`, `metrics.md`, `scenarios.md`, `perturbations.md`, `schemas.md`) are **MkDocs sources** in git (mkdocstrings / mkdocs-macros). Jekyll’s **Liquid** parser does not understand that syntax, so a plain `bundle exec jekyll serve` fails on `models.md`.

CI runs `docs/scripts/jekyll_prepare_mkdocstring_pages.py` before Jekyll to expand those files. Do the same locally from the **repository root**:

```bash
pip install -r docs/requirements.txt   # once: MkDocs, mkdocstrings, beautifulsoup4, html2text, …
chmod +x docs/scripts/serve_jekyll_local.sh   # once, if needed
./docs/scripts/serve_jekyll_local.sh
```

Open http://localhost:4000/ — when you stop the server (Ctrl+C), the script restores the original `docs/*.md` sources from git so nothing is left expanded by mistake.

### Jekyll only (home and non-MkDocs pages)

If you only need a quick look at layouts like the homepage and do not care that `/models/` and other reference pages are missing or broken, you can still run `cd docs && bundle exec jekyll serve` after temporarily moving aside `models.md`, or rely on the script above for a faithful build.

(Alternatively, use [rbenv](https://github.com/rbenv/rbenv) or [asdf](https://asdf-vm.com/) to install Ruby 3 or 4.)

Jekyll does not use the Python venv; it uses Bundler (Ruby). For MedHELM Python setup, see [Getting started from a git clone](#getting-started-from-a-git-clone-development) above or the [repository README](../README.md).

