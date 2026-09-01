"""HealthAdminBench Client: one HELM Request is one Playwright episode."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import yaml

from helm.benchmark.model_deployment_registry import ModelDeployment, get_model_deployment
from helm.benchmark.scenarios.health_admin_bench_constants import (
    ACTION_SPACES,
    DEFAULT_ENV_BASE_URL,
    HAB_HARNESS_DEPLOYMENT,
    HAB_PROTOCOL,
    HAB_ROOT_ENV,
    OBSERVATION_MODES,
    PROMPT_MODES,
)
from helm.benchmark.scenarios.health_admin_bench_scenario import resolve_hab_root
from helm.clients.auto_client import AutoClient
from helm.clients.client import Client
from helm.clients.health_admin_bench_helm_agent import (
    create_helm_backed_agent,
    is_openai_chat_compatible,
    make_judge_complete,
)
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog, hwarn
from helm.common.request import ErrorFlags, GeneratedOutput, Request, RequestResult, Token

# Playwright's sync API and `_hab_runtime`'s `os.chdir` are process-global.
# MedHELM `--num-threads` otherwise runs several `make_request()` calls at once.
_HAB_EPISODE_LOCK = threading.Lock()


def load_model_map() -> Dict[str, Any]:
    map_path = Path(__file__).resolve().parents[1] / "benchmark" / "static" / "health_admin_bench_model_map.yaml"
    with map_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def lookup_model_mapping(evaluated_model: str, evaluated_deployment: str = "") -> Optional[Dict[str, Any]]:
    """Return a HAB map override, or None to try MedHELM helm_backed."""
    data = load_model_map()
    entries = data.get("models") or []
    exact = None
    prefix_match = None
    prefix_len = -1
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if evaluated_deployment and entry.get("model_deployment") == evaluated_deployment:
            return entry
        if entry.get("model") == evaluated_model:
            exact = entry
            continue
        prefix = entry.get("model_prefix")
        if prefix and evaluated_model.startswith(prefix) and len(prefix) > prefix_len:
            prefix_match = entry
            prefix_len = len(prefix)
    if exact is not None:
        return exact
    if prefix_match is not None:
        return prefix_match
    return None


@contextmanager
def _hab_runtime(hab_root: Path) -> Iterator[None]:
    prev_cwd = Path.cwd()
    env_file = hab_root / ".env"
    if str(hab_root) not in sys.path:
        sys.path.insert(0, str(hab_root))
    os.chdir(hab_root)
    try:
        from dotenv import load_dotenv

        if env_file.is_file():
            load_dotenv(env_file, override=False)
        yield
    finally:
        os.chdir(prev_cwd)


def _empty_completion_payload(error: str, envelope: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": envelope.get("task_id"),
        "passed": False,
        "score": 0.0,
        "max_points": 0.0,
        "percentage": 0.0,
        "steps": 0,
        "eval_results": [],
        "evaluated_model": envelope.get("evaluated_model"),
        "evaluated_model_deployment": envelope.get("evaluated_model_deployment"),
        "hab_agent": None,
        "prompt_mode": envelope.get("prompt_mode"),
        "observation_mode": envelope.get("observation_mode"),
        "error": error,
    }


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _try_get_deployment(name: str) -> Optional[ModelDeployment]:
    if not name:
        return None
    try:
        return get_model_deployment(name)
    except ValueError:
        return None


def _require_allowed(name: str, value: Any, allowed: tuple[str, ...]) -> str:
    """Strip and lowercase ``value``; raise with the allow-list if it is not valid."""
    raw = "" if value is None else str(value)
    normalized = raw.strip().lower()
    if normalized not in allowed:
        raise ValueError(f"HealthAdminBench invalid {name}={raw!r}; allowed: {'|'.join(allowed)}")
    return normalized


def _strip_identity(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_evaluated_identity(envelope: Dict[str, Any], request_model: str = "") -> tuple[str, str]:
    """Require a named evaluated model before backend routing.

    Fills a missing deployment from the model (and vice versa). Does not use the
    outer Request.model_deployment, which is always ``hab/harness``.
    """
    evaluated_model = _strip_identity(envelope.get("evaluated_model"))
    evaluated_deployment = _strip_identity(envelope.get("evaluated_model_deployment"))
    fallback_model = _strip_identity(request_model)
    if fallback_model == HAB_HARNESS_DEPLOYMENT:
        fallback_model = ""
    if not evaluated_model and fallback_model:
        evaluated_model = fallback_model

    evaluated_deployment = evaluated_deployment or evaluated_model
    evaluated_model = evaluated_model or evaluated_deployment

    if not evaluated_model and not evaluated_deployment:
        raise ValueError(
            "HealthAdminBench envelope missing evaluated_model and evaluated_model_deployment; "
            "set model= and model_deployment= on the run entry."
        )
    if evaluated_model == HAB_HARNESS_DEPLOYMENT or evaluated_deployment == HAB_HARNESS_DEPLOYMENT:
        raise ValueError("evaluated_model_deployment must not be hab/harness")

    envelope["evaluated_model"] = evaluated_model
    envelope["evaluated_model_deployment"] = evaluated_deployment
    return evaluated_model, evaluated_deployment


class HealthAdminBenchClient(Client):
    """Runs HealthAdminBench `run_task` in-process. Does not cache GUI episodes."""

    def __init__(
        self,
        cache_config: Optional[CacheConfig] = None,
        model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        auto_client: Optional[AutoClient] = None,
    ):
        del cache_config, tokenizer_name
        self.model_name = model_name or HAB_HARNESS_DEPLOYMENT
        self.auto_client = auto_client

    def make_request(self, request: Request) -> RequestResult:
        started = time.time()
        try:
            envelope = json.loads(request.prompt)
        except json.JSONDecodeError as exc:
            return self._failed_result(str(exc), {}, started)

        if envelope.get("hab_protocol") != HAB_PROTOCOL:
            return self._failed_result(
                f"Expected hab_protocol={HAB_PROTOCOL}",
                envelope,
                started,
            )

        try:
            _normalize_evaluated_identity(envelope, request.model)
            with _HAB_EPISODE_LOCK:
                payload = self._run_episode(envelope)
            text = json.dumps(payload)
            return RequestResult(
                success=True,
                cached=False,
                request_time=time.time() - started,
                request_datetime=int(time.time()),
                completions=[GeneratedOutput(text=text, logprob=0, tokens=[Token(text=text, logprob=0)])],
                embedding=[],
            )
        except Exception as exc:  # noqa: BLE001 - episode failures must not abort the suite
            hwarn(f"HealthAdminBench episode failed: {exc}")
            return self._failed_result(str(exc), envelope, started)

    def _failed_result(self, error: str, envelope: Dict[str, Any], started: float) -> RequestResult:
        text = json.dumps(_empty_completion_payload(error, envelope))
        return RequestResult(
            success=False,
            cached=False,
            error=error,
            error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
            request_time=time.time() - started,
            request_datetime=int(time.time()),
            completions=[GeneratedOutput(text=text, logprob=0, tokens=[Token(text=text, logprob=0)])],
            embedding=[],
        )

    def _resolve_backend(
        self, evaluated_model: str, evaluated_deployment: str
    ) -> tuple[str, Optional[Dict[str, Any]], Optional[ModelDeployment]]:
        mapping = lookup_model_mapping(evaluated_model, evaluated_deployment)
        backend = (mapping or {}).get("backend") or ""
        if mapping and backend == "hab_native":
            return "hab_native", mapping, None

        deployment_name = evaluated_deployment or evaluated_model
        if deployment_name == HAB_HARNESS_DEPLOYMENT:
            raise ValueError("evaluated_model_deployment must not be hab/harness")
        deployment = _try_get_deployment(deployment_name)
        if deployment is None and evaluated_model:
            deployment = _try_get_deployment(evaluated_model)
        if mapping and backend not in (None, "", "hab_native", "helm_backed"):
            raise ValueError(f"Unsupported HealthAdminBench backend: {backend!r}")
        if deployment is not None:
            if not is_openai_chat_compatible(deployment):
                raise ValueError(
                    f"HealthAdminBench helm_backed requires an OpenAI-chat-compatible client, "
                    f"got {deployment.client_spec.class_name} for {deployment.name}. "
                    "Add a hab_native map row or use OpenAIClient / GrokChatClient."
                )
            return "helm_backed", mapping, deployment
        raise ValueError(
            f"HealthAdminBench has no agent for model {evaluated_model!r} "
            f"(deployment {evaluated_deployment!r}); add a map entry in "
            "health_admin_bench_model_map.yaml or an OpenAI-compatible model_deployment."
        )

    def _judge_complete(self, envelope: Dict[str, Any]) -> Optional[Callable[[str], str]]:
        judge_model = str(envelope.get("judge_model") or "")
        judge_deployment = str(envelope.get("judge_model_deployment") or "")
        if not judge_deployment or self.auto_client is None:
            return None
        if judge_deployment == HAB_HARNESS_DEPLOYMENT:
            raise ValueError(
                f"HealthAdminBench judge must not use {HAB_HARNESS_DEPLOYMENT!r}; "
                "set jury_config_path or judge_model_deployment"
            )
        return make_judge_complete(
            self.auto_client,
            judge_model or judge_deployment,
            judge_deployment,
            HAB_HARNESS_DEPLOYMENT,
        )

    def _run_episode(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        hab_root = resolve_hab_root(str(envelope.get("hab_root") or os.environ.get(HAB_ROOT_ENV) or ""))
        evaluated_model = str(envelope.get("evaluated_model") or "")
        evaluated_deployment = str(envelope.get("evaluated_model_deployment") or "")
        backend, mapping, deployment = self._resolve_backend(evaluated_model, evaluated_deployment)

        prompt_mode = _require_allowed("prompt_mode", envelope.get("prompt_mode") or "general", PROMPT_MODES)
        observation_mode = _require_allowed(
            "observation_mode",
            envelope.get("observation_mode") or (mapping or {}).get("observation_mode_default") or "axtree_only",
            OBSERVATION_MODES,
        )
        action_space = _require_allowed(
            "action_space",
            envelope.get("action_space") or (mapping or {}).get("action_space_default") or "dom",
            ACTION_SPACES,
        )
        envelope["prompt_mode"] = prompt_mode
        envelope["observation_mode"] = observation_mode
        envelope["action_space"] = action_space
        env_base_url = envelope.get("env_base_url") or DEFAULT_ENV_BASE_URL
        is_gui = _as_bool(envelope.get("is_gui"), False)
        max_steps = envelope.get("max_steps")
        if max_steps is not None and max_steps != "":
            max_steps = int(max_steps)
        else:
            max_steps = None

        task_relpath = envelope.get("task_relpath")
        if not task_relpath:
            raise ValueError("HealthAdminBench envelope missing task_relpath")
        task_path = hab_root / task_relpath
        hab_model_id = (mapping or {}).get("hab_model_id") or evaluated_model
        llm_complete = self._judge_complete(envelope)

        hlog(
            f"HealthAdminBench episode task={envelope.get('task_id')} "
            f"evaluated_model={evaluated_model} backend={backend} "
            f"hab_model={hab_model_id} obs={observation_mode} max_steps={max_steps} "
            f"judge={envelope.get('judge_model_deployment') or 'hab-fallback'} "
            f"is_gui={is_gui}"
        )

        with _hab_runtime(hab_root):
            from harness.prompts import ActionSpace, ObservationMode, PromptMode
            from run import run_task

            prompt_mode_map = {
                "zero_shot": PromptMode.ZERO_SHOT,
                "general": PromptMode.GENERAL,
                "task_specific": PromptMode.TASK_SPECIFIC,
            }
            observation_mode_map = {
                "screenshot_only": ObservationMode.SCREENSHOT_ONLY,
                "axtree_only": ObservationMode.AXTREE_ONLY,
                "both": ObservationMode.BOTH,
            }
            injected_agent = None
            if backend == "helm_backed":
                if self.auto_client is None:
                    raise ValueError("HelmBackedAgent requires AutoClient injection")
                assert deployment is not None
                injected_agent = create_helm_backed_agent(
                    auto_client=self.auto_client,
                    deployment=deployment,
                    prompt_mode=prompt_mode_map[str(prompt_mode)],
                    observation_mode=observation_mode_map[str(observation_mode)],
                    action_space=ActionSpace(str(action_space)),
                )
            result = run_task(
                model=hab_model_id if backend == "hab_native" else (deployment.name if deployment else hab_model_id),
                task_file=str(task_path),
                env_base_url=env_base_url,
                max_steps=max_steps,
                is_gui=is_gui,
                prompt_mode=prompt_mode_map[str(prompt_mode)],
                observation_mode=observation_mode_map[str(observation_mode)],
                action_space=ActionSpace(str(action_space)),
                agent=injected_agent,
                llm_complete=llm_complete,
            )

        payload = result.to_dict()
        payload.update(
            {
                "steps": getattr(result, "steps", payload.get("steps", 0)),
                "evaluated_model": evaluated_model,
                "evaluated_model_deployment": evaluated_deployment,
                "hab_agent": getattr(result, "agent_name", (mapping or {}).get("agent_class")),
                "prompt_mode": prompt_mode,
                "observation_mode": observation_mode,
                "judge_model": envelope.get("judge_model"),
                "judge_model_deployment": envelope.get("judge_model_deployment"),
                "error": None,
            }
        )
        return payload
