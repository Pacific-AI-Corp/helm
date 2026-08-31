import json
import sys
from pathlib import Path

import pytest

from helm.benchmark.model_deployment_registry import ClientSpec, ModelDeployment
from helm.benchmark.scenarios.health_admin_bench_constants import HAB_HARNESS_DEPLOYMENT
from helm.clients.health_admin_bench_helm_agent import (
    HAB_JUDGE_SYSTEM,
    is_openai_chat_compatible,
    make_judge_complete,
    step_max_tokens,
)
from helm.common.request import GeneratedOutput, Request, RequestResult


HAB_ROOT = Path(__file__).resolve().parents[5] / "health-admin-bench"


class FakeAutoClient:
    def __init__(self, text: str):
        self.text = text
        self.requests: list = []

    def make_request(self, request: Request) -> RequestResult:
        self.requests.append(request)
        return RequestResult(
            success=True,
            cached=False,
            completions=[GeneratedOutput(text=self.text, logprob=0, tokens=[])],
            embedding=[],
        )


def _openai_deployment(name: str = "openai/gpt-4o-2024-05-13") -> ModelDeployment:
    return ModelDeployment(
        name=name,
        client_spec=ClientSpec(class_name="helm.clients.openai_client.OpenAIClient"),
        model_name=name,
        max_sequence_length=128000,
    )


def test_is_openai_chat_compatible_openai_and_grok():
    assert is_openai_chat_compatible(_openai_deployment())
    grok = ModelDeployment(
        name="xai/grok-4-0709",
        client_spec=ClientSpec(class_name="helm.clients.grok_client.GrokChatClient"),
        model_name="xai/grok-4-0709",
        max_sequence_length=256000,
    )
    assert is_openai_chat_compatible(grok)
    simple = ModelDeployment(
        name="simple/model1",
        client_spec=ClientSpec(class_name="helm.clients.simple_client.SimpleClient"),
        model_name="simple/model1",
        max_sequence_length=2048,
    )
    assert not is_openai_chat_compatible(simple)


def test_step_max_tokens_from_window():
    assert step_max_tokens(_openai_deployment()) == 4096
    short = ModelDeployment(
        name="local/tiny",
        client_spec=ClientSpec(class_name="helm.clients.openai_client.OpenAIClient"),
        model_name="local/tiny",
        max_sequence_length=2048,
    )
    assert step_max_tokens(short) == 256


def test_make_judge_complete_uses_judge_deployment_not_harness():
    fake = FakeAutoClient('{"score": 1, "reasoning": "ok", "evidence_quote": "note"}')
    complete = make_judge_complete(
        fake,  # type: ignore[arg-type]
        "openai/gpt-4o-2024-05-13",
        "openai/gpt-4o-2024-05-13",
        HAB_HARNESS_DEPLOYMENT,
    )
    text = complete("grade this submission")
    assert '"score": 1' in text
    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request.model_deployment == "openai/gpt-4o-2024-05-13"
    assert request.model_deployment != HAB_HARNESS_DEPLOYMENT
    assert request.messages[0]["role"] == "system"
    assert HAB_JUDGE_SYSTEM in request.messages[0]["content"]
    assert request.messages[1]["content"] == "grade this submission"


def test_make_judge_complete_rejects_harness_deployment():
    complete = make_judge_complete(
        FakeAutoClient(""),  # type: ignore[arg-type]
        "hab/harness",
        HAB_HARNESS_DEPLOYMENT,
        HAB_HARNESS_DEPLOYMENT,
    )
    with pytest.raises(ValueError, match="must not use"):
        complete("prompt")


def test_resolve_backend_native_and_helm_backed(monkeypatch):
    from helm.clients.health_admin_bench_client import HealthAdminBenchClient

    openai_dep = _openai_deployment()
    simple_dep = ModelDeployment(
        name="simple/model1",
        client_spec=ClientSpec(class_name="helm.clients.simple_client.SimpleClient"),
        model_name="simple/model1",
        max_sequence_length=2048,
    )
    registry = {openai_dep.name: openai_dep, simple_dep.name: simple_dep}

    def fake_get(name: str, warn_deprecated: bool = False) -> ModelDeployment:
        if name not in registry:
            raise ValueError(f"Model deployment {name} not found")
        return registry[name]

    monkeypatch.setattr("helm.clients.health_admin_bench_client.get_model_deployment", fake_get)
    client = HealthAdminBenchClient()

    backend, mapping, deployment = client._resolve_backend("simple/model1", "simple/model1")
    assert backend == "hab_native"
    assert mapping is not None
    assert mapping["agent_class"] == "RandomAgent"
    assert deployment is None

    backend, mapping, deployment = client._resolve_backend(
        "openai/gpt-4o-2024-05-13", "openai/gpt-4o-2024-05-13"
    )
    assert backend == "helm_backed"
    assert mapping is None
    assert deployment is openai_dep

    backend, mapping, _ = client._resolve_backend(
        "anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-5-sonnet-20241022"
    )
    assert backend == "hab_native"

    with pytest.raises(ValueError, match="must not be hab/harness"):
        client._resolve_backend("x", HAB_HARNESS_DEPLOYMENT)

    with pytest.raises(ValueError, match="no agent"):
        client._resolve_backend("unknown/model", "unknown/model")


def test_resolve_backend_fails_closed_for_simple_client_as_evaluated_deployment(monkeypatch):
    from helm.clients.health_admin_bench_client import HealthAdminBenchClient

    simple_dep = ModelDeployment(
        name="vllm/not-chat",
        client_spec=ClientSpec(class_name="helm.clients.simple_client.SimpleClient"),
        model_name="vllm/not-chat",
        max_sequence_length=2048,
    )

    def fake_get(name: str, warn_deprecated: bool = False) -> ModelDeployment:
        if name != simple_dep.name:
            raise ValueError(f"Model deployment {name} not found")
        return simple_dep

    monkeypatch.setattr("helm.clients.health_admin_bench_client.get_model_deployment", fake_get)
    client = HealthAdminBenchClient()
    with pytest.raises(ValueError, match="OpenAI-chat-compatible"):
        client._resolve_backend("vllm/not-chat", "vllm/not-chat")


def test_helm_backed_agent_parses_action_from_fake_client():
    if not HAB_ROOT.is_dir():
        pytest.skip("health-admin-bench checkout not found")
    if str(HAB_ROOT) not in sys.path:
        sys.path.insert(0, str(HAB_ROOT))

    from harness.prompts import ActionSpace, ObservationMode, PromptMode

    from helm.clients.health_admin_bench_helm_agent import create_helm_backed_agent

    fake = FakeAutoClient("ACTION: scroll(down)\nKEY_INFO: looking around")
    agent = create_helm_backed_agent(
        auto_client=fake,  # type: ignore[arg-type]
        deployment=_openai_deployment(),
        prompt_mode=PromptMode.GENERAL,
        observation_mode=ObservationMode.AXTREE_ONLY,
        action_space=ActionSpace.DOM,
    )
    action = agent.get_action(
        {
            "axtree_txt": "RootWebArea 'EMR'",
            "goal": "Open the worklist",
            "url": "https://emrportal.vercel.app/worklist",
            "title": "Worklist",
            "step": 1,
        }
    )
    assert action == "scroll(down)"
    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request.model_deployment == "openai/gpt-4o-2024-05-13"
    assert request.model_deployment != HAB_HARNESS_DEPLOYMENT
    assert request.messages[0]["role"] == "system"
    assert request.messages[1]["role"] == "user"
    assert request.temperature == 0.0


def test_health_admin_bench_spec_judge_override(tmp_path):
    from helm.benchmark.run_specs.medhelm_run_specs import get_health_admin_bench_spec

    judges = tmp_path / "judges.yaml"
    judges.write_text(
        json.dumps(
            {
                "judges": [
                    {
                        "name": "gpt",
                        "model": "openai/gpt-4o-2024-05-13",
                        "model_deployment": "openai/gpt-4o-2024-05-13",
                    }
                ]
            }
        ).replace("'", '"'),
        encoding="utf-8",
    )
    # yaml.safe_load accepts JSON
    spec = get_health_admin_bench_spec(
        task_ids="emr-easy-1",
        jury_config_path=str(judges),
    )
    knobs = json.loads(spec.adapter_spec.instructions)
    assert knobs["judge_model"] == "openai/gpt-4o-2024-05-13"
    assert knobs["judge_model_deployment"] == "openai/gpt-4o-2024-05-13"
    assert knobs["jury_config_path"] == str(judges)

    override = get_health_admin_bench_spec(
        judge_model="xai/grok-4-0709",
        judge_model_deployment="xai/grok-4-0709",
    )
    override_knobs = json.loads(override.adapter_spec.instructions)
    assert override_knobs["judge_model"] == "xai/grok-4-0709"
    assert override_knobs["judge_model_deployment"] == "xai/grok-4-0709"
