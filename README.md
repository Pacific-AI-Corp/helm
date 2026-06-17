# MedHELM

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PacificAI/medhelm/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/medhelm?color=blue)](https://pypi.org/project/medhelm/)

<img src="https://github.com/PacificAI/medhelm/raw/main/docs/assets/images/medhelm_logo.jpg" alt="MedHELM" width="320"/>

## MedHELM

**MedHELM** is a multi-institutional effort to develop standardized, clinically grounded benchmarks for evaluating large language models in healthcare. While it builds on the HELM evaluation framework, MedHELM is independently developed through a broad collaboration spanning **Stanford Medicine**, **HAI**, **Microsoft**, and partners across the healthcare and research ecosystem.

The initiative focuses on real-world clinical tasks, emphasizing:
  - Transparency
  - Reproducibility
  - Practical relevance for healthcare deployment

This framework includes the following features:

  - Datasets and benchmarks in a standardized format (e.g. MMLU-Pro, GPQA, IFEval, WildBench)
  - Models from various providers accessible through a unified interface (e.g. OpenAI models, Anthropic Claude, Google Gemini)
  - Metrics for measuring various aspects beyond accuracy (e.g. efficiency, bias, toxicity)
  - Web UI for inspecting individual prompts and responses
  - Web leaderboard for comparing results across models and benchmarks

## Documentation

Documentation: **[medhelm.org](https://medhelm.org)**

## Install & run (MedHELM library)

MedHELM uses the HELM core engine and adds medical benchmarks.

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
medhelm-run --run-entries "pubmed_qa:model=openai/gpt2,model_deployment=huggingface/gpt2" --suite my_med_test --max-eval-instances 2 --num-threads 1
helm-summarize --suite my_med_test -o ./benchmark_output
helm-server --suite my_med_test -o ./benchmark_output --port 8000
```

Open **http://localhost:8000** in your browser (not `0.0.0.0`). Press `Ctrl-C` in the terminal to stop the server.

Local Hugging Face models (e.g. `gpt2`) automatically use the best available PyTorch device: **CUDA → Apple MPS → CPU**.

---

### Getting started from PyPI (no git clone)

If you only want to run benchmarks and view results — no local code changes:

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install medhelm
```

The PyPI package includes a pre-built web UI (no Node.js required). Then run the commands in step 5 below.

### Optional tiers (summarization & gated)

The standard install (`uv pip install -e .` or `uv pip install medhelm`) covers PubMedQA, MedCalc-Bench, MedicationQA, and MedHallu. For clinical summarization or gated (Google Drive) scenarios, install the extra dependencies **once**:

```sh
# From a git clone (editable install):
uv pip install -e ".[summarization,gated]"

# From PyPI:
uv pip install "medhelm[summarization,gated]"
```

The summarization extra adds bert-score, rouge-score, and nltk (**2–3 minute** install). The gated extra adds **gdown** for MedQA / MedMCQA dataset downloads.

### Standard tier (recommended to start)

Scenarios: **PubMedQA**, **MedCalc-Bench**, **MedicationQA**, **MedHallu**.

Included in the base install — no extra `[summarization]` or `[gated]` needed.

**Quick test** (small local model, 2 instances — runs in seconds):

```sh
medhelm-run --run-entries "pubmed_qa:model=openai/gpt2,model_deployment=huggingface/gpt2" --suite my_med_test --max-eval-instances 2 --num-threads 1
helm-summarize --suite my_med_test -o ./benchmark_output
helm-server --suite my_med_test -o ./benchmark_output --port 8000
```

**Full example** (better quality, 10 instances; needs more RAM/VRAM):

```sh
medhelm-run --run-entries "pubmed_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" --suite my_med_test --max-eval-instances 10
helm-summarize --suite my_med_test -o ./benchmark_output
helm-server --suite my_med_test -o ./benchmark_output --port 8000
```

Then open http://localhost:8000/ in your browser.

### Clinical NLP tier (`[summarization]`)

Scenarios: **ACI-Bench** (clinical transcripts; no extra data required), **Patient-Edu** (simplifying medical jargon), **DischargeMe** (hospital course summaries; requires PhysioNet credentials and a `data_path`).

Install (in addition to the base install):

```sh
# Git clone:
uv pip install -e ".[summarization]"

# PyPI:
uv pip install "medhelm[summarization]"
```

**Quick test** — ACI-Bench with `gpt2`, 2 instances (first run may take several minutes while metrics download):

```sh
medhelm-run --run-entries "aci_bench:model=openai/gpt2,model_deployment=huggingface/gpt2" --suite med_summaries --max-eval-instances 2 --num-threads 1
helm-summarize --suite med_summaries -o ./benchmark_output
helm-server --suite med_summaries -o ./benchmark_output --port 8000
```

**Full example** — ACI-Bench with a larger model:

```sh
medhelm-run --run-entries "aci_bench:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" --suite med_summaries --max-eval-instances 5
helm-summarize --suite med_summaries -o ./benchmark_output
helm-server --suite med_summaries -o ./benchmark_output --port 8000
```

**Notes:**
- Prefer **ACI-Bench** for a first summarization test — it does not require external data files.
- **DischargeMe** is not a quick test; it needs a PhysioNet `data_path` argument in the run entry.
- The first ACI-Bench run downloads evaluation assets (e.g. QAFactEval) and runs heavyweight metrics — expect minutes, not seconds.
- ACI-Bench uses an LLM jury for some scores; the default jury config targets hosted APIs. Local runs complete, but jury scores may be empty without API credentials.

### Gated / licensing tier (`[gated]`)

Adds **gdown** for scenarios that download datasets from Google Drive.

Scenarios: **MedQA** (USMLE/Board exams), **MedMCQA** (AIIMS/NEET exams).

Install (in addition to the base install):

```sh
# Git clone:
uv pip install -e ".[gated]"

# PyPI:
uv pip install "medhelm[gated]"
```

**Quick test** — MedQA with `gpt2`, 2 instances (downloads the dataset from Google Drive on first run):

```sh
medhelm-run --run-entries "med_qa:model=openai/gpt2,model_deployment=huggingface/gpt2" --suite board_exams --max-eval-instances 2 --num-threads 1
helm-summarize --suite board_exams -o ./benchmark_output
helm-server --suite board_exams -o ./benchmark_output --port 8000
```

**Full example** — MedQA with a larger model:

```sh
medhelm-run --run-entries "med_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" --suite board_exams --max-eval-instances 10
helm-summarize --suite board_exams -o ./benchmark_output
helm-server --suite board_exams -o ./benchmark_output --port 8000
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|--------|----------------|-----|
| `command not found: uv` | uv not installed or not on `PATH` | [Install uv](#0-install-tools-once-per-machine), then `source $HOME/.local/bin/env` |
| `ModuleNotFoundError: helm.benchmark.static_build` | Web UI not built (git clone only) | [Build the web UI](#4-build-the-web-ui-required-before-helm-server) |
| `command not found: npm` | Node.js not installed | Install Node.js 18+ (only needed for `helm-server` after a git clone) |
| `zsh: bus error` on Mac when running a local Hugging Face model | PyTorch CPU crash on Apple Silicon | Use a build with MPS device support (CUDA → MPS → CPU); add `--num-threads 1` |
| Blank browser page at `helm-server` | Wrong URL or UI not built | Open **http://localhost:8000** (not `0.0.0.0`); rebuild the UI if needed |
| Missing `gdown` / Drive download errors | Gated extra not installed | `uv pip install -e ".[gated]"` or `uv pip install "medhelm[gated]"` |
| Summarization import / metric errors | Summarization extra not installed | `uv pip install -e ".[summarization]"` or `uv pip install "medhelm[summarization]"` |
| Run runs out of memory | Model too large (e.g. Qwen 7B) | Use the **quick test** commands with `gpt2` first |

**Tips:**
- Always run `helm-summarize` after `medhelm-run`, then `helm-server`.
- On macOS, add `--num-threads 1` to `medhelm-run` if you see instability with local models.
- Local Hugging Face models use the best available device: **CUDA → Apple MPS → CPU**.

### Alternative: Using pip

If you prefer `pip` instead of `uv`:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                              # git clone
# pip install medhelm                         # PyPI
# pip install -e ".[summarization,gated]"     # optional tiers (git clone)
```

### Classic HELM commands

You can still use `helm-run`, `helm-summarize`, and `helm-server`; `medhelm-run` is an alias for `helm-run`.

After activating your environment:

```sh
medhelm-run --run-entries mmlu:subject=philosophy,model=openai/gpt2 --suite my-suite --max-eval-instances 10
helm-summarize --suite my-suite -o ./benchmark_output
helm-server --suite my-suite -o ./benchmark_output --port 8000
```

## Quick Start (summary)

<!--quick-start-begin-->

| Tier | Install | Scenarios |
|------|--------|-----------|
| **Standard** | `uv pip install -e .` (repo) or `uv pip install medhelm` (PyPI) | PubMedQA, MedCalc-Bench, MedicationQA, MedHallu |
| **Summarization** | `uv pip install -e ".[summarization]"` or `uv pip install "medhelm[summarization]"` | ACI-Bench, Patient-Edu, DischargeMe (2–3 min install) |
| **Gated** | `uv pip install -e ".[gated]"` or `uv pip install "medhelm[gated]"` | MedQA, MedMCQA (Google Drive) |
| **All optional tiers** | `uv pip install -e ".[summarization,gated]"` or `uv pip install "medhelm[summarization,gated]"` | All of the above |

**Quick tests** (local `gpt2`, 2 instances): `pubmed_qa` (standard), `aci_bench` (summarization), `med_qa` (gated). Always run `helm-summarize` then `helm-server` after `medhelm-run`. See [Install & run](#install--run-medhelm-library) for full commands and [Troubleshooting](#troubleshooting).

<!--quick-start-end-->

## Goals & roadmap

MedHELM aims to be a **new public repo** with **fewer dependencies**, **easier installation**, and **public documentation**. We welcome feedback on the following:

- **HealthBench:** We are considering new subcategories to include HealthBench. Do you see value in adding HealthBench, and how would you use it?
- **Non-gated alternatives:** We provide **7 non-gated datasets** (e.g. PubMedQA, MedCalc-Bench, MedicationQA, MedHallu, and others in the Standard and Summarization tiers) as free alternatives for the same kinds of tasks as gated benchmarks.
- **Hospital & private data:** We want to make it **easier for hospital systems to contribute or add their own private datasets**. If your institution is interested in running or contributing benchmarks, we’d like to hear from you.

## Leaderboard

We maintain a **medical** leaderboard for comparing models on MedHELM benchmarks:

- **[MedHELM Leaderboard](https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard)** — PubMedQA, MedQA, MedMCQA, and other medical benchmarks.

To reproduce or extend results locally, see [medhelm.org](https://medhelm.org) (Reproducing Leaderboards, MedHELM docs).

## Citation

MedHELM builds on the Holistic Evaluation of Language Models framework. If you use this software in your research, please cite the MedHELM and HELM papers as below.

```bibtex

@Article{Bedi2026,
author={Bedi, Suhana and Cui, Hejie and Fuentes, Miguel and Unell, Alyssa and Wornow, Michael and Banda, Juan M. and Kotecha, Nikesh and Keyes, Timothy and Mai, Yifan and Oez, Mert and Qiu, Hao and Jain, Shrey and Schettini, Leonardo and Kashyap, Mehr and Fries, Jason Alan and Swaminathan, Akshay and Chung, Philip and Haredasht, Fateme Nateghi and Lopez, Ivan and Aali, Asad and Tse, Gabriel and Nayak, Ashwin and Vedak, Shivam and Jain, Sneha S. and Patel, Birju and Fayanju, Oluseyi and Shah, Shreya and Goh, Ethan and Yao, Dong-han and Soetikno, Brian and Reis, Eduardo and Gatidis, Sergios and Divi, Vasu and Capasso, Robson and Saralkar, Rachna and Chiang, Chia-Chun and Jindal, Jenelle and Pham, Tho and Ghoddusi, Faraz and Lin, Steven and Chiou, Albert S. and Hong, Hyo Jung and Roy, Mohana and Gensheimer, Michael F. and Patel, Hinesh and Schulman, Kevin and Dash, Dev and Char, Danton and Downing, Lance and Grolleau, Francois and Black, Kameron and Mieso, Bethel and Zahedivash, Aydin and Yim, Wen-wai and Sharma, Harshita and Lee, Tony and Kirsch, Hannah and Lee, Jennifer and Ambers, Nerissa and Lugtu, Carlene and Sharma, Aditya and Mawji, Bilal and Alekseyev, Alex and Zhou, Vicky and Kakkar, Vikas and Helzer, Jarrod and Revri, Anurang and Bannett, Yair and Daneshjou, Roxana and Chen, Jonathan and Alsentzer, Emily and Morse, Keith and Ravi, Nirmal and Aghaeepour, Nima and Kennedy, Vanessa and Chaudhari, Akshay and Wang, Thomas and Koyejo, Sanmi and Lungren, Matthew P. and Horvitz, Eric and Liang, Percy and Pfeffer, Michael A. and Shah, Nigam H.},
title={Holistic evaluation of large language models for medical tasks with MedHELM},
journal={Nature Medicine},
year={2026},
month={Mar},
day={01},
volume={32},
number={3},
pages={943-951},
abstract={While large language models (LLMs) achieve near-perfect scores on medical licensing exams, these evaluations inadequately reflect the complexity and diversity of real-world clinical practice. Here we introduce MedHELM, an extensible evaluation framework with three contributions. First, a clinician-validated taxonomy organizing medical AI applications into five categories that mirror real clinical tasks---clinical decision support (diagnostic decisions, treatment planning), clinical note generation (visit documentation, procedure reports), patient communication (education materials, care instructions), medical research (literature analysis, clinical data analysis) and administration (scheduling, workflow coordination). These encompass 22 subcategories and 121 specific tasks reflecting daily medical practice. Second, a comprehensive benchmark suite of 37 evaluations covering all subcategories. Third, systematic comparison of nine frontier LLMs---Claude 3.5 Sonnet, Claude 3.7 Sonnet, DeepSeek R1, Gemini 1.5 Pro, Gemini 2.0 Flash, GPT-4o, GPT-4o mini, Llama 3.3 and o3-mini---using an automated LLM-jury evaluation method. Our LLM-jury uses multiple AI evaluators to assess model outputs against expert-defined criteria. Advanced reasoning models (DeepSeek R1, o3-mini) demonstrated superior performance with win rates of 66{\%}, although Claude 3.5 Sonnet achieved comparable results at 15{\%} lower computational cost. These results not only highlight current model capabilities but also demonstrate how MedHELM could enable evidence-based selection of medical AI systems for healthcare applications.},
issn={1546-170X},
doi={10.1038/s41591-025-04151-2},
url={https://doi.org/10.1038/s41591-025-04151-2}
}
```
