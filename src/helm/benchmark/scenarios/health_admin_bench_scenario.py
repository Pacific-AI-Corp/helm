"""HealthAdminBench scenario: one MedHELM instance per GUI admin task."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.health_admin_bench_constants import (
    DEFAULT_TASK_VERSION,
    DIFFICULTIES,
    HAB_PROTOCOL,
    HAB_ROOT_ENV,
    TASK_SETS,
)
from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Input,
    Instance,
    Scenario,
    ScenarioMetadata,
)


def resolve_hab_root(explicit: str = "") -> Path:
    """Locate the HealthAdminBench checkout."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_root = os.environ.get(HAB_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root).expanduser())
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "health-admin-bench",
            cwd.parent / "health-admin-bench",
        ]
    )
    # health_admin_bench_scenario.py -> scenarios -> benchmark -> helm -> src -> medhelm -> workspace
    here = Path(__file__).resolve()
    if len(here.parents) >= 6:
        candidates.append(here.parents[5] / "health-admin-bench")
    if len(here.parents) >= 5:
        candidates.append(here.parents[4].parent / "health-admin-bench")

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "benchmark" / "v2" / "tasks").is_dir():
            return resolved

    raise FileNotFoundError(
        "Could not find HealthAdminBench. Set HEALTH_ADMIN_BENCH_ROOT or pass hab_root= "
        "to the health_admin_bench run spec."
    )


def parse_task_ids(task_ids: str) -> List[str]:
    """Parse a '+'-separated task id list from a run entry (commas are reserved)."""
    if not task_ids or not str(task_ids).strip():
        return []
    return [part.strip() for part in str(task_ids).replace(",", "+").split("+") if part.strip()]


class HealthAdminBenchScenario(Scenario):
    """Wrap HealthAdminBench v2 task JSON files as MedHELM instances."""

    name = "health_admin_bench"
    description = (
        "HealthAdminBench evaluates computer-use agents on synthetic healthcare administration "
        "workflows (prior authorization, appeals/denials, and DME order processing) against "
        "simulated EMR, payer, and fax portals."
    )
    tags = ["health", "administration", "agentic", "computer-use"]

    def __init__(
        self,
        hab_root: str = "",
        task_set: str = "prior_auth",
        difficulty: str = "easy",
        version: str = DEFAULT_TASK_VERSION,
        task_ids: str = "",
        prompt_mode: str = "general",
        observation_mode: str = "axtree_only",
        action_space: str = "dom",
        env_base_url: str = "https://emrportal.vercel.app",
        max_steps: str = "",
    ):
        super().__init__()
        self.hab_root = hab_root
        self.task_set = (task_set or "prior_auth").strip()
        self.difficulty = (difficulty or "easy").strip().lower()
        self.version = (version or DEFAULT_TASK_VERSION).strip()
        self.task_ids = parse_task_ids(task_ids)
        self.prompt_mode = prompt_mode or "general"
        self.observation_mode = observation_mode or "axtree_only"
        self.action_space = action_space or "dom"
        self.env_base_url = env_base_url or "https://emrportal.vercel.app"
        self.max_steps = str(max_steps).strip() if max_steps is not None else ""

        if self.task_set not in TASK_SETS and self.task_set != "all":
            raise ValueError(f"task_set must be one of {TASK_SETS + ('all',)}, got {self.task_set!r}")
        if self.difficulty not in DIFFICULTIES and self.difficulty != "all":
            raise ValueError(f"difficulty must be one of {DIFFICULTIES + ('all',)}, got {self.difficulty!r}")

    def _task_set_dirs(self) -> Iterable[str]:
        if self.task_set == "all":
            return TASK_SETS
        return (self.task_set,)

    def _iter_task_files(self, hab_root: Path) -> List[Path]:
        files: List[Path] = []
        tasks_root = hab_root / "benchmark" / self.version / "tasks"
        if not tasks_root.is_dir():
            raise FileNotFoundError(f"HealthAdminBench tasks not found: {tasks_root}")
        for task_set in self._task_set_dirs():
            files.extend(sorted((tasks_root / task_set).glob("*.json")))
        return files

    def get_instances(self, output_path: str) -> List[Instance]:
        del output_path  # tasks live in the HAB checkout, not HELM's scenario cache
        hab_root = resolve_hab_root(self.hab_root)
        allowed_ids = set(self.task_ids)
        instances: List[Instance] = []

        for task_path in self._iter_task_files(hab_root):
            with task_path.open("r", encoding="utf-8") as handle:
                task = json.load(handle)
            task_id = str(task.get("id") or task_path.stem)
            if allowed_ids and task_id not in allowed_ids:
                continue
            difficulty = str(task.get("difficulty", "")).lower()
            if self.difficulty != "all" and difficulty != self.difficulty:
                continue

            relpath = task_path.relative_to(hab_root).as_posix()
            parent_name = task_path.parent.name
            evals = task.get("evals") or []
            envelope = {
                "hab_protocol": HAB_PROTOCOL,
                "task_id": task_id,
                "task_relpath": relpath,
                "goal": task.get("goal", ""),
                "difficulty": difficulty,
                "task_set": parent_name,
                "challengeType": task.get("challengeType") or task.get("category") or "",
                "website_id": (task.get("website") or {}).get("id", ""),
                "prompt_mode": self.prompt_mode,
                "observation_mode": self.observation_mode,
                "action_space": self.action_space,
                "env_base_url": self.env_base_url,
                "hab_root": str(hab_root),
            }
            if self.max_steps:
                envelope["max_steps"] = int(self.max_steps)

            extra_data = {
                "task_relpath": relpath,
                "difficulty": difficulty,
                "task_set": parent_name,
                "challengeType": envelope["challengeType"],
                "max_points": sum(float(item.get("points", 0)) for item in evals),
                "n_jmespath": sum(1 for item in evals if item.get("type") == "jmespath"),
                "n_llm_judge": sum(1 for item in evals if item.get("type") == "llm_judge"),
            }
            instances.append(
                Instance(
                    id=task_id,
                    input=Input(text=json.dumps(envelope)),
                    references=[],
                    split=TEST_SPLIT,
                    extra_data=extra_data,
                )
            )

        if not instances:
            raise ValueError(
                f"No HealthAdminBench tasks matched task_set={self.task_set!r}, "
                f"difficulty={self.difficulty!r}, task_ids={self.task_ids!r} under {hab_root}"
            )
        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="health_admin_bench",
            display_name="HealthAdminBench",
            description=(
                "HealthAdminBench evaluates computer-use agents on 135 synthetic healthcare "
                "administration tasks across EMR, payer, and fax portals "
                "[(Bedi et al., 2026)](https://arxiv.org/abs/2604.09937)."
            ),
            taxonomy=TaxonomyInfo(
                task="Computer-use agent evaluation",
                what="Complete prior authorization, appeals/denials, and DME workflows in simulated portals",
                when="Any",
                who="Hospital administrator",
                language="English",
            ),
            main_metric="health_admin_bench_score",
            main_split="test",
        )
