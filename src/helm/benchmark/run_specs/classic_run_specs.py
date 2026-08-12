"""Run spec functions for the HELM Classic leaderboard.

Website: https://crfm.stanford.edu/helm/classic/

If a run spec function is included in both the HELM Classic leaderboard and the
HELM Lite leaderboard, it will be included in the lite_run_specs module instead of this module.
This module also contains some scenarios that are currently not used on any HELM leaderboard."""

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_MULTIPLE_CHOICE_JOINT,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_multiple_choice_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_exact_match_metric_specs,
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("truthful_qa")
def get_truthful_qa_spec(task: str, method: str = ADAPT_MULTIPLE_CHOICE_JOINT) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.truthful_qa_scenario.TruthfulQAScenario",
        args={"task": task},
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=method, instructions="", input_noun="Question", output_noun="Answer"
    )

    return RunSpec(
        name=f"truthful_qa:task={task},method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["truthful_qa"],
    )
