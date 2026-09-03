import json
import threading
import time

import pytest

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, ADAPT_HEALTH_ADMIN_BENCH
from helm.benchmark.runner import execute_parallelism_for_run
from helm.benchmark.scenarios.health_admin_bench_constants import (
    ACTION_SPACES,
    HAB_HARNESS_DEPLOYMENT,
    HAB_PROTOCOL,
    OBSERVATION_MODES,
    PROMPT_MODES,
)
from helm.clients.health_admin_bench_client import (
    HealthAdminBenchClient,
    _normalize_evaluated_identity,
    _require_allowed,
)
from helm.common.cache import SqliteCacheConfig
from helm.common.cache_backend_config import SqliteCacheBackendConfig
from helm.common.request import Request


def test_execute_parallelism_for_hab_is_clamped_to_one():
    assert execute_parallelism_for_run(ADAPT_HEALTH_ADMIN_BENCH, 4) == 1
    assert execute_parallelism_for_run(ADAPT_HEALTH_ADMIN_BENCH, 1) == 1
    assert execute_parallelism_for_run(ADAPT_GENERATION, 4) == 4


def test_hab_client_serializes_concurrent_episodes(monkeypatch):
    client = HealthAdminBenchClient()
    current = 0
    max_seen = 0
    lock = threading.Lock()
    start_together = threading.Barrier(2)

    def fake_run(self, envelope):
        nonlocal current, max_seen
        with lock:
            current += 1
            max_seen = max(max_seen, current)
        time.sleep(0.08)
        with lock:
            current -= 1
        return {
            "task_id": envelope.get("task_id"),
            "passed": False,
            "score": 0.0,
            "max_points": 0.0,
            "percentage": 0.0,
            "steps": 0,
            "eval_results": [],
        }

    monkeypatch.setattr(HealthAdminBenchClient, "_run_episode", fake_run)
    prompt = json.dumps(
        {
            "hab_protocol": HAB_PROTOCOL,
            "task_id": "emr-easy-1",
            "evaluated_model": "simple/model1",
            "evaluated_model_deployment": "simple/model1",
        }
    )
    request = Request(prompt=prompt, model_deployment=HAB_HARNESS_DEPLOYMENT)

    results: list = []

    def worker():
        start_together.wait(timeout=5)
        results.append(client.make_request(request))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert max_seen == 1
    assert len(results) == 2
    assert all(result.success for result in results)


def test_normalize_evaluated_identity_fills_missing_side():
    envelope = {"evaluated_model": "openai/gpt-4o-2024-05-13"}
    model, deployment = _normalize_evaluated_identity(envelope)
    assert model == "openai/gpt-4o-2024-05-13"
    assert deployment == "openai/gpt-4o-2024-05-13"
    assert envelope["evaluated_model_deployment"] == "openai/gpt-4o-2024-05-13"

    envelope = {"evaluated_model_deployment": "xai/grok-4-0709"}
    model, deployment = _normalize_evaluated_identity(envelope)
    assert model == "xai/grok-4-0709"
    assert deployment == "xai/grok-4-0709"
    assert envelope["evaluated_model"] == "xai/grok-4-0709"


def test_normalize_evaluated_identity_rejects_empty_and_whitespace():
    with pytest.raises(ValueError, match="missing evaluated_model and evaluated_model_deployment"):
        _normalize_evaluated_identity({})
    with pytest.raises(ValueError, match="missing evaluated_model and evaluated_model_deployment"):
        _normalize_evaluated_identity({"evaluated_model": "  ", "evaluated_model_deployment": ""})


def test_normalize_evaluated_identity_rejects_hab_harness():
    with pytest.raises(ValueError, match="must not be hab/harness"):
        _normalize_evaluated_identity({"evaluated_model_deployment": HAB_HARNESS_DEPLOYMENT})
    with pytest.raises(ValueError, match="must not be hab/harness"):
        _normalize_evaluated_identity({"evaluated_model": HAB_HARNESS_DEPLOYMENT})


def test_normalize_evaluated_identity_uses_request_model_not_harness_deployment():
    envelope: dict = {}
    model, deployment = _normalize_evaluated_identity(envelope, request_model="openai/gpt-4o-2024-05-13")
    assert model == deployment == "openai/gpt-4o-2024-05-13"
    with pytest.raises(ValueError, match="missing evaluated_model and evaluated_model_deployment"):
        _normalize_evaluated_identity({}, request_model=HAB_HARNESS_DEPLOYMENT)


def test_make_request_missing_identity_is_not_generic_no_agent(monkeypatch):
    client = HealthAdminBenchClient()

    def fail_run(self, envelope):
        raise AssertionError("_run_episode should not run when identity is missing")

    monkeypatch.setattr(HealthAdminBenchClient, "_run_episode", fail_run)
    result = client.make_request(
        Request(
            prompt=json.dumps({"hab_protocol": HAB_PROTOCOL, "task_id": "emr-easy-1"}),
            model_deployment=HAB_HARNESS_DEPLOYMENT,
        )
    )
    assert not result.success
    assert "missing evaluated_model and evaluated_model_deployment" in (result.error or "")
    assert "no agent" not in (result.error or "")


def test_make_request_fills_deployment_from_model_before_episode(monkeypatch):
    client = HealthAdminBenchClient()
    seen: dict = {}

    def fake_run(self, envelope):
        seen.update(envelope)
        return {"task_id": envelope.get("task_id"), "passed": False, "score": 0.0, "eval_results": []}

    monkeypatch.setattr(HealthAdminBenchClient, "_run_episode", fake_run)
    result = client.make_request(
        Request(
            prompt=json.dumps(
                {
                    "hab_protocol": HAB_PROTOCOL,
                    "task_id": "emr-easy-1",
                    "evaluated_model": "simple/model1",
                }
            ),
            model_deployment=HAB_HARNESS_DEPLOYMENT,
        )
    )
    assert result.success
    assert seen["evaluated_model"] == "simple/model1"
    assert seen["evaluated_model_deployment"] == "simple/model1"


def test_require_allowed_accepts_valid_and_casefold():
    assert _require_allowed("observation_mode", "axtree_only", OBSERVATION_MODES) == "axtree_only"
    assert _require_allowed("observation_mode", "AXTREE_ONLY", OBSERVATION_MODES) == "axtree_only"
    assert _require_allowed("prompt_mode", " general ", PROMPT_MODES) == "general"
    assert _require_allowed("action_space", "DOM", ACTION_SPACES) == "dom"


def test_require_allowed_lists_allowed_values():
    with pytest.raises(
        ValueError, match=r"invalid observation_mode='axtree'; allowed: screenshot_only\|axtree_only\|both"
    ):
        _require_allowed("observation_mode", "axtree", OBSERVATION_MODES)
    with pytest.raises(ValueError, match=r"invalid prompt_mode='few_shot'; allowed: zero_shot\|general\|task_specific"):
        _require_allowed("prompt_mode", "few_shot", PROMPT_MODES)
    with pytest.raises(ValueError, match=r"invalid action_space='mouse'; allowed: dom\|coordinate"):
        _require_allowed("action_space", "mouse", ACTION_SPACES)


def test_sqlite_cache_path_survives_hab_chdir(tmp_path, monkeypatch):
    cache_dir = tmp_path / "prod_env" / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    backend = SqliteCacheBackendConfig("prod_env/cache")
    elsewhere = tmp_path / "health-admin-bench"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cache_config = backend.get_cache_config("azure")
    assert isinstance(cache_config, SqliteCacheConfig)
    assert cache_config.path == str(cache_dir / "azure.sqlite")
