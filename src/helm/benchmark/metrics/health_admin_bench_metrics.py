"""Metrics for HealthAdminBench episode scores."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog


def parse_hab_completion(request_state: RequestState) -> Optional[Dict[str, Any]]:
    if request_state.result is None or not request_state.result.completions:
        return None
    text = request_state.result.completions[0].text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        hlog("Warning: HealthAdminBench completion was not valid JSON")
        return None
    return payload if isinstance(payload, dict) else None


def _ratio(points: float, max_points: float) -> float:
    if max_points <= 0:
        return 0.0
    return points / max_points


def _typed_scores(eval_results: List[Dict[str, Any]], eval_type: str) -> float:
    earned = 0.0
    maximum = 0.0
    for row in eval_results:
        if row.get("type") != eval_type:
            continue
        earned += float(row.get("points") or 0.0)
        maximum += float(row.get("max_points") or 0.0)
    return _ratio(earned, maximum)


class HealthAdminBenchMetric(Metric):
    """Unpack HAB EvaluationResult JSON from the episode completion."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        del adapter_spec, metric_service, eval_cache_path
        payload = parse_hab_completion(request_state)
        if not payload:
            hlog("Warning: No HealthAdminBench completion payload; recording zeros")
            payload = {}

        percentage = float(payload.get("percentage") or 0.0)
        score = float(payload.get("score") or 0.0)
        max_points = float(payload.get("max_points") or 0.0)
        passed = 1.0 if payload.get("passed") else 0.0
        steps = float(payload.get("steps") or 0.0)
        eval_results = payload.get("eval_results") or []
        if not isinstance(eval_results, list):
            eval_results = []

        return [
            Stat(MetricName("health_admin_bench_score")).add(percentage / 100.0),
            Stat(MetricName("health_admin_bench_pass")).add(passed),
            Stat(MetricName("health_admin_bench_points")).add(score),
            Stat(MetricName("health_admin_bench_max_points")).add(max_points),
            Stat(MetricName("health_admin_bench_jmespath_score")).add(_typed_scores(eval_results, "jmespath")),
            Stat(MetricName("health_admin_bench_llm_judge_score")).add(_typed_scores(eval_results, "llm_judge")),
            Stat(MetricName("health_admin_bench_steps")).add(steps),
        ]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="health_admin_bench_score",
                display_name="HealthAdminBench Score",
                short_display_name="HAB Score",
                description=(
                    "Fraction of HealthAdminBench subeval points earned (JMESPath plus LLM-judge), "
                    "normalized to 0–1."
                ),
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="health_admin_bench_pass",
                display_name="HealthAdminBench Pass Rate",
                short_display_name="HAB Pass",
                description="1 if the task earned 100% of subeval points, else 0.",
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="health_admin_bench_points",
                display_name="HealthAdminBench Points",
                short_display_name="HAB Points",
                description="Raw subeval points earned on the task.",
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="health_admin_bench_max_points",
                display_name="HealthAdminBench Max Points",
                short_display_name="HAB Max",
                description="Maximum subeval points available on the task.",
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="health_admin_bench_jmespath_score",
                display_name="HealthAdminBench JMESPath Score",
                short_display_name="HAB JMESPath",
                description="Fraction of deterministic JMESPath subeval points earned.",
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="health_admin_bench_llm_judge_score",
                display_name="HealthAdminBench LLM-Judge Score",
                short_display_name="HAB Judge",
                description="Fraction of LLM-judge subeval points earned.",
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="health_admin_bench_steps",
                display_name="HealthAdminBench Steps",
                short_display_name="HAB Steps",
                description="Number of browser actions taken during the episode.",
                lower_is_better=True,
                group=None,
            ),
        ]
