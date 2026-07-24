---
title: Code Structure
---
# Code Structure

Overview of the MedHELM benchmarking pipeline and how to extend it.

## Birds-eye view

- A **`Scenario`** (from a `ScenarioSpec`) defines a task and dataset. It produces **`Instance`** objects with inputs and **`Reference`** outputs.
- A **`DataPreprocessor`** converts a scenario into instances and applies augmentations from `DataAugmenterSpec`.
- An **`Adapter`** (from `AdaptationSpec`) turns instances into **`Request`** objects for the model.
- An **`Executor`** runs requests and collects **`RequestResult`** objects in a **`ScenarioState`**.
- A **`Metric`** (from `MetricSpec`) computes **`Stat`** objects (accuracy, ROUGE, summarization scores, etc.).
- A **`Runner`** orchestrates the above for each **`RunSpec`**.

Class categories:

- **Specifications** (`RunSpec`, `AdapterSpec`, …) — user configuration
- **States** (`Instance`, `RequestResult`, …) — serializable data
- **Controllers** (`Scenario`, `Adapter`, `Metric`, `Runner`) — implementation logic

MedHELM run specs live in `src/helm/benchmark/run_specs/`. Scenarios live in `src/helm/benchmark/scenarios/`.

## Adding new scenarios

1. Create `src/helm/benchmark/scenarios/your_scenario.py` with a `Scenario` subclass implementing `get_instances()`.
2. Each `Instance` needs `Input`, `Reference`(s), and a split (`TRAIN_SPLIT`, `VALID_SPLIT`, or `TEST_SPLIT`). Mark correct references with `CORRECT_TAG`.
3. Set `name`, `description`, and `tags` on the scenario class.
4. Choose metrics in an existing `*_metrics.py` or add a task-specific metric class. Many tasks use `basic_metrics.py` via `common_metric_specs.py`.
5. Add a `@run_spec_function("your_name")` in `src/helm/benchmark/run_specs/medhelm_run_specs.py` that builds `ScenarioSpec`, `AdapterSpec`, `MetricSpec` list, and returns a `RunSpec`.
6. Test with `medhelm-run -r your_name:model=openai/gpt2 --suite dev --max-eval-instances 10`.
7. For leaderboard visibility, add the scenario to `src/helm/benchmark/static/schema_medhelm.yaml`.

For private organization data, read from a configured local path (see private run entry configs such as `run_entries_medhelm_private_stanford.conf`) rather than committing restricted files.

See [Adding New Scenarios](/adding_new_scenarios) for a step-by-step tutorial with MedHELM examples.

## Adding new metrics

**Generic metrics** (reusable across tasks):

1. Add a scoring function to `basic_metrics.py` or `evaluate_reference_metrics.py`.
2. Register it in the metric function mapping.
3. Expose it via `common_metric_specs.py` if needed.

**Task-specific metrics:**

1. Create `your_task_metrics.py` with a class extending `Metric` from `metric.py`.
2. Implement `evaluate_generation()` returning a list of `Stat` objects.

Summarization benchmarks use `summarization_metrics.py` (requires `pip install "medhelm[summarization]"`).

## Data augmentations

Pass a `DataAugmenterSpec` with `PerturbationSpec` entries into `RunSpec`:

```python
data_augmenter_spec = DataAugmenterSpec(
    perturbation_specs=[
        PerturbationSpec(
            class_name="helm.benchmark.augmentations.perturbation.ExtraSpacePerturbation",
            args={"num_spaces": 5},
        )
    ],
    should_perturb_references=False,
    should_augment_train_instances=False,
    should_include_original_train=False,
    should_augment_eval_instances=True,
    should_include_original_eval=True,
)
```

See [Perturbations](/perturbations) for available perturbations.

## Multimodal MedHELM benchmarks

Image-generation scenarios (radiology, mental disorders) live under `src/helm/benchmark/scenarios/image_generation/` with metrics under `src/helm/benchmark/metrics/image_generation/`. Install `pip install "medhelm[heim]"` for image-generation metrics.

Vision-language (VQA-RAD) and audio (speech disorder) scenarios have dedicated run spec modules: `medical_multimodal_run_specs.py` and `speech_disorder_audio_run_specs.py`.

## Supporting new Hugging Face tokenizers

1. Use the Hugging Face model name (e.g. `EleutherAI/gpt-j-6B`).
2. Add loading logic in `HuggingFaceTokenizers.load_tokenizer`.
3. Add a test in `test_huggingface_tokenizer.py`.
4. Add a `WindowService` subclass and register it in `WindowServiceFactory`.

See [Adding New Tokenizers](/adding_new_tokenizers).
