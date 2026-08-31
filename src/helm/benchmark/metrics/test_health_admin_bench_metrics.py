import json

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.health_admin_bench_metrics import HealthAdminBenchMetric
from helm.benchmark.scenarios.scenario import TEST_SPLIT, Input, Instance
from helm.common.request import GeneratedOutput, Request, RequestResult


def _request_state(payload: dict) -> RequestState:
    instance = Instance(input=Input(text=""), references=[], split=TEST_SPLIT, id="emr-easy-1")
    return RequestState(
        instance=instance,
        reference_index=None,
        request_mode=None,
        train_trial_index=0,
        output_mapping=None,
        request=Request(prompt="unused"),
        result=RequestResult(
            success=True,
            cached=False,
            completions=[GeneratedOutput(text=json.dumps(payload), logprob=0, tokens=[])],
            embedding=[],
        ),
        num_train_instances=0,
        prompt_truncated=False,
    )


def test_health_admin_bench_metric_parses_episode_json():
    payload = {
        "task_id": "emr-easy-1",
        "passed": False,
        "score": 2.0,
        "max_points": 4.0,
        "percentage": 50.0,
        "steps": 12,
        "eval_results": [
            {"type": "jmespath", "success": True, "points": 1.0, "max_points": 1.0},
            {"type": "jmespath", "success": False, "points": 0.0, "max_points": 1.0},
            {"type": "llm_judge", "success": True, "points": 1.0, "max_points": 1.0},
            {"type": "llm_judge", "success": False, "points": 0.0, "max_points": 1.0},
        ],
    }
    stats = {
        stat.name.name: stat.mean
        for stat in HealthAdminBenchMetric().evaluate_generation(
            AdapterSpec(),
            _request_state(payload),
            None,  # type: ignore[arg-type]
            "",
        )
    }
    assert stats["health_admin_bench_score"] == 0.5
    assert stats["health_admin_bench_pass"] == 0.0
    assert stats["health_admin_bench_points"] == 2.0
    assert stats["health_admin_bench_max_points"] == 4.0
    assert stats["health_admin_bench_jmespath_score"] == 0.5
    assert stats["health_admin_bench_llm_judge_score"] == 0.5
    assert stats["health_admin_bench_steps"] == 12.0


def test_health_admin_bench_metric_zeros_on_empty_completion():
    instance = Instance(input=Input(text=""), references=[], split=TEST_SPLIT, id="x")
    request_state = RequestState(
        instance=instance,
        reference_index=None,
        request_mode=None,
        train_trial_index=0,
        output_mapping=None,
        request=Request(prompt=""),
        result=RequestResult(success=False, cached=False, completions=[], embedding=[]),
        num_train_instances=0,
        prompt_truncated=False,
    )
    stats = {
        stat.name.name: stat.mean
        for stat in HealthAdminBenchMetric().evaluate_generation(
            AdapterSpec(),
            request_state,
            None,  # type: ignore[arg-type]
            "",
        )
    }
    assert stats["health_admin_bench_score"] == 0.0
    assert stats["health_admin_bench_pass"] == 0.0
