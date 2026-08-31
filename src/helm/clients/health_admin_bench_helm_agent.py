"""HelmBackedAgent: HAB BaseAgent whose get_action() calls MedHELM AutoClient."""

from __future__ import annotations

from typing import Any, Dict

from helm.benchmark.model_deployment_registry import ModelDeployment
from helm.benchmark.scenarios.health_admin_bench_constants import HAB_HARNESS_DEPLOYMENT
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog, hwarn
from helm.common.request import Request, RequestResult

HAB_JUDGE_SYSTEM = (
    "You are a grader. Return strict JSON with keys "
    "score, reasoning, evidence_quote (score must be 0 or 1). "
    "Use only evidence from <STUDENT_SUBMISSION>."
)

OPENAI_CHAT_CLIENT_NAMES = (
    "helm.clients.openai_client.OpenAIClient",
    "helm.clients.grok_client.GrokChatClient",
)


def is_openai_chat_compatible(deployment: ModelDeployment) -> bool:
    class_name = (deployment.client_spec.class_name or "").strip()
    if class_name in OPENAI_CHAT_CLIENT_NAMES:
        return True
    try:
        from helm.clients.openai_client import OpenAIClient
        from helm.common.object_spec import get_class_by_name

        cls = get_class_by_name(class_name)
        return issubclass(cls, OpenAIClient)
    except Exception:  # noqa: BLE001
        return False


def step_max_tokens(deployment: ModelDeployment) -> int:
    window = deployment.max_sequence_length or 8192
    return min(4096, max(256, window // 8))


def make_judge_complete(
    auto_client: AutoClient,
    judge_model: str,
    judge_deployment: str,
    harness_deployment: str,
) -> Any:
    def complete(prompt: str) -> str:
        if not judge_deployment or judge_deployment == harness_deployment:
            raise ValueError(
                f"HealthAdminBench judge must not use {harness_deployment!r}; "
                "set jury_config_path or judge_model_deployment"
            )
        hlog(f"HealthAdminBench judge via {judge_deployment} ({judge_model})")
        result: RequestResult = auto_client.make_request(
            Request(
                model=judge_model,
                model_deployment=judge_deployment,
                messages=[
                    {"role": "system", "content": HAB_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
                num_completions=1,
            )
        )
        if not result.success or not result.completions:
            raise RuntimeError(result.error or "HealthAdminBench MedHELM judge request failed")
        return result.completions[0].text

    return complete


def create_helm_backed_agent(
    auto_client: AutoClient,
    deployment: ModelDeployment,
    prompt_mode: Any,
    observation_mode: Any,
    action_space: Any,
):
    """Build a HAB BaseAgent subclass after HAB is on sys.path."""
    if deployment.name == HAB_HARNESS_DEPLOYMENT:
        raise ValueError(
            f"HelmBackedAgent must not use {HAB_HARNESS_DEPLOYMENT!r} as the evaluated deployment"
        )
    from harness.agents.base import BaseAgent
    from harness.prompts import get_prompt_builder

    class HelmBackedAgent(BaseAgent):
        def __init__(self) -> None:
            super().__init__(name=f"HelmBackedAgent({deployment.name})")
            self.prompt_mode = prompt_mode
            self.observation_mode = observation_mode
            self.action_space = action_space
            self.last_actions: list = []
            self.last_observations: list = []
            self.api_failures = 0
            self.max_api_failures = 3
            self.prompt_builder = get_prompt_builder(prompt_mode, action_space=action_space)
            self._deployment = deployment
            self._auto_client = auto_client

        def get_action(self, observation: Dict[str, Any]) -> str:
            base_prompt = self.convert_observation_to_base_prompt(
                observation,
                last_actions=self.last_actions,
                last_observations=self.last_observations,
                is_screenshot_available=False,
                observation_mode=self.observation_mode,
                prompt_builder=self.prompt_builder,
            )
            system_msg = base_prompt["system_msg"]
            user_msg = base_prompt["user_msg"]
            step = base_prompt["step"]
            max_tokens = step_max_tokens(self._deployment)
            hlog(
                f"HealthAdminBench HelmBackedAgent step={step} "
                f"deployment={self._deployment.name} max_tokens={max_tokens}"
            )
            result = self._auto_client.make_request(
                Request(
                    model=self._deployment.model_name or self._deployment.name,
                    model_deployment=self._deployment.name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    num_completions=1,
                )
            )
            if not result.success or not result.completions:
                self.api_failures += 1
                hwarn(
                    f"HelmBackedAgent API failure {self.api_failures}/{self.max_api_failures}: "
                    f"{result.error}"
                )
                if self.api_failures >= self.max_api_failures:
                    raise RuntimeError(f"HelmBackedAgent failed: {result.error}")
                action = "scroll(down)"
                self.last_actions.append(action)
                self.last_observations.append("API failure")
                return action

            self.api_failures = 0
            text = result.completions[0].text or ""
            parsed = self.prompt_builder.extract_response_fields(text)
            action = parsed["action"]
            key_info = parsed["key_info"]
            self.set_step_trace(
                model_action=action,
                model_key_info=key_info,
                model_thinking=parsed.get("thinking", ""),
                model_raw_response=parsed.get("raw_response", text),
            )
            self.last_actions.append(action)
            self.last_observations.append(key_info)
            return action

    return HelmBackedAgent()
