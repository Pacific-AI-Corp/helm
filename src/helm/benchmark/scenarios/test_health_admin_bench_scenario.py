import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.health_admin_bench_adapter import build_hab_request
from helm.benchmark.scenarios.health_admin_bench_constants import (
    HAB_HARNESS_DEPLOYMENT,
    HAB_PROTOCOL,
    HAB_ROOT_ENV,
)
from helm.benchmark.scenarios.health_admin_bench_scenario import HealthAdminBenchScenario, resolve_hab_root
from helm.benchmark.scenarios.scenario import TEST_SPLIT, Input, Instance


MINIMAL_TASK = {
    "id": "emr-easy-1",
    "goal": "Open referral REF-2025-002 and document whether prior auth is required.",
    "website": {"id": "emr", "name": "EMR Referral Portal", "url": "https://emrportal.vercel.app"},
    "difficulty": "easy",
    "challengeType": "no_auth_medicare",
    "possible": True,
    "evals": [
        {
            "type": "jmespath",
            "query": "full_state.agentActions.addedAuthNote",
            "points": 1,
            "expected_value": True,
        },
        {
            "type": "llm_judge",
            "description": "Note quality",
            "student_answer": "{{full_state.communications[-1].content}}",
            "student_answer_context": "note",
            "rubric": "Score 1 if the note is correct.",
            "points": 1,
        },
    ],
    "config": {"task_id": "easy_1", "start_url": "/worklist"},
    "version": "v2",
}


def _write_fixture(root: Path) -> Path:
    task_dir = root / "benchmark" / "v2" / "tasks" / "prior_auth"
    task_dir.mkdir(parents=True)
    path = task_dir / "emr-easy-1.json"
    path.write_text(json.dumps(MINIMAL_TASK), encoding="utf-8")
    return path


def test_health_admin_bench_scenario_get_instances_parses_envelope_and_extra_data():
    with TemporaryDirectory() as tmpdir:
        hab_root = Path(tmpdir)
        _write_fixture(hab_root)
        scenario = HealthAdminBenchScenario(
            hab_root=str(hab_root),
            task_set="prior_auth",
            difficulty="easy",
            task_ids="emr-easy-1",
            prompt_mode="general",
            observation_mode="axtree_only",
            action_space="dom",
            max_steps="3",
        )
        instances = scenario.get_instances(str(hab_root / "out"))

    assert len(instances) == 1
    instance = instances[0]
    assert instance.id == "emr-easy-1"
    assert instance.split == TEST_SPLIT
    envelope = json.loads(instance.input.text)
    assert envelope["hab_protocol"] == HAB_PROTOCOL
    assert envelope["task_id"] == "emr-easy-1"
    assert envelope["task_relpath"] == "benchmark/v2/tasks/prior_auth/emr-easy-1.json"
    assert envelope["difficulty"] == "easy"
    assert envelope["task_set"] == "prior_auth"
    assert envelope["challengeType"] == "no_auth_medicare"
    assert envelope["website_id"] == "emr"
    assert envelope["max_steps"] == 3
    assert instance.extra_data is not None
    assert instance.extra_data["max_points"] == 2.0
    assert instance.extra_data["n_jmespath"] == 1
    assert instance.extra_data["n_llm_judge"] == 1


def test_health_admin_bench_scenario_filters_task_ids():
    with TemporaryDirectory() as tmpdir:
        hab_root = Path(tmpdir)
        _write_fixture(hab_root)
        extra = hab_root / "benchmark" / "v2" / "tasks" / "prior_auth" / "emr-easy-2.json"
        extra.write_text(json.dumps({**MINIMAL_TASK, "id": "emr-easy-2"}), encoding="utf-8")
        scenario = HealthAdminBenchScenario(
            hab_root=str(hab_root),
            task_set="prior_auth",
            difficulty="easy",
            task_ids="emr-easy-1",
        )
        instances = scenario.get_instances(str(hab_root / "out"))
    assert [instance.id for instance in instances] == ["emr-easy-1"]


def test_lookup_model_mapping_exact_and_prefix():
    from helm.clients.health_admin_bench_client import lookup_model_mapping

    random_map = lookup_model_mapping("simple/model1")
    assert random_map is not None
    assert random_map["hab_model_id"] == "random"
    assert random_map["backend"] == "hab_native"
    # OpenAI chat models are helm_backed (no map row)
    assert lookup_model_mapping("openai/gpt-4o-2024-05-13") is None
    claude = lookup_model_mapping("anthropic/claude-3-5-sonnet-20241022")
    assert claude is not None
    assert claude["hab_model_id"] == "claude-opus-4-6"
    assert lookup_model_mapping("unknown/model") is None


def test_build_hab_request_rewrites_model_deployment_and_injects_evaluated_model():
    envelope = {
        "hab_protocol": HAB_PROTOCOL,
        "task_id": "emr-easy-1",
        "task_relpath": "benchmark/v2/tasks/prior_auth/emr-easy-1.json",
        "goal": "do the task",
    }
    instance = Instance(input=Input(text=json.dumps(envelope)), references=[], split=TEST_SPLIT, id="emr-easy-1")
    adapter_spec = AdapterSpec(
        method="health_admin_bench",
        model="openai/gpt-4o-2024-05-13",
        model_deployment="openai/gpt-4o-2024-05-13",
        instructions=json.dumps({"prompt_mode": "general", "max_steps": 3}),
        max_train_instances=0,
        max_tokens=1,
        temperature=0.0,
        num_outputs=1,
    )
    request = build_hab_request(instance, adapter_spec)
    assert request.model_deployment == HAB_HARNESS_DEPLOYMENT
    assert request.model == "openai/gpt-4o-2024-05-13"
    payload = json.loads(request.prompt)
    assert payload["evaluated_model"] == "openai/gpt-4o-2024-05-13"
    assert payload["evaluated_model_deployment"] == "openai/gpt-4o-2024-05-13"
    assert payload["prompt_mode"] == "general"
    assert payload["max_steps"] == 3


def test_build_hab_request_copies_is_gui():
    envelope = {
        "hab_protocol": HAB_PROTOCOL,
        "task_id": "emr-easy-1",
        "task_relpath": "benchmark/v2/tasks/prior_auth/emr-easy-1.json",
        "goal": "do the task",
    }
    instance = Instance(input=Input(text=json.dumps(envelope)), references=[], split=TEST_SPLIT, id="emr-easy-1")
    adapter_spec = AdapterSpec(
        method="health_admin_bench",
        model="openai/gpt-4o-2024-05-13",
        model_deployment="openai/gpt-4o-2024-05-13",
        instructions=json.dumps({"is_gui": True}),
        max_train_instances=0,
        max_tokens=1,
        temperature=0.0,
        num_outputs=1,
    )
    payload = json.loads(build_hab_request(instance, adapter_spec).prompt)
    assert payload["is_gui"] is True


def test_build_hab_request_copies_judge_knobs():
    envelope = {
        "hab_protocol": HAB_PROTOCOL,
        "task_id": "emr-easy-1",
        "task_relpath": "benchmark/v2/tasks/prior_auth/emr-easy-1.json",
        "goal": "do the task",
    }
    instance = Instance(input=Input(text=json.dumps(envelope)), references=[], split=TEST_SPLIT, id="emr-easy-1")
    adapter_spec = AdapterSpec(
        method="health_admin_bench",
        model="openai/gpt-4o-2024-05-13",
        model_deployment="openai/gpt-4o-2024-05-13",
        instructions=json.dumps(
            {
                "judge_model": "openai/gpt-4o-2024-05-13",
                "judge_model_deployment": "openai/gpt-4o-2024-05-13",
                "jury_config_path": "src/helm/benchmark/static/health_admin_bench_judges.yaml",
            }
        ),
        max_train_instances=0,
        max_tokens=1,
        temperature=0.0,
        num_outputs=1,
    )
    payload = json.loads(build_hab_request(instance, adapter_spec).prompt)
    assert payload["judge_model"] == "openai/gpt-4o-2024-05-13"
    assert payload["judge_model_deployment"] == "openai/gpt-4o-2024-05-13"
    assert payload["jury_config_path"].endswith("health_admin_bench_judges.yaml")


def test_resolve_hab_root_explicit_wins_over_env(monkeypatch, tmp_path):
    chosen = tmp_path / "chosen"
    decoy = tmp_path / "decoy"
    _write_fixture(chosen)
    _write_fixture(decoy)
    monkeypatch.setenv(HAB_ROOT_ENV, str(decoy))
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert resolve_hab_root(str(chosen)) == chosen.resolve()


def test_resolve_hab_root_uses_env_when_explicit_empty(monkeypatch, tmp_path):
    hab = tmp_path / "hab"
    _write_fixture(hab)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv(HAB_ROOT_ENV, str(hab))
    monkeypatch.chdir(work)
    assert resolve_hab_root("") == hab.resolve()


def test_resolve_hab_root_cwd_and_parent_siblings(monkeypatch, tmp_path):
    monkeypatch.delenv(HAB_ROOT_ENV, raising=False)

    cwd_parent = tmp_path / "cwd_case"
    cwd_parent.mkdir()
    hab_cwd = cwd_parent / "health-admin-bench"
    _write_fixture(hab_cwd)
    monkeypatch.chdir(cwd_parent)
    assert resolve_hab_root("") == hab_cwd.resolve()

    parent_case = tmp_path / "parent_case"
    work = parent_case / "medhelm"
    work.mkdir(parents=True)
    hab_parent = parent_case / "health-admin-bench"
    _write_fixture(hab_parent)
    monkeypatch.chdir(work)
    assert resolve_hab_root("") == hab_parent.resolve()


def test_resolve_hab_root_missing_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv(HAB_ROOT_ENV, raising=False)
    work = tmp_path / "nowhere"
    work.mkdir()
    monkeypatch.chdir(work)
    with pytest.raises(FileNotFoundError, match="HEALTH_ADMIN_BENCH_ROOT") as exc_info:
        resolve_hab_root("")
    assert "hab_root=" in str(exc_info.value)
    assert "benchmark/v2/tasks" in str(exc_info.value)
