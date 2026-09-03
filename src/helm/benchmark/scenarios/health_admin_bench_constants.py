"""Shared constants for the HealthAdminBench MedHELM integration."""

HAB_PROTOCOL = "health_admin_bench.v1"
HAB_HARNESS_DEPLOYMENT = "hab/harness"
HAB_ROOT_ENV = "HEALTH_ADMIN_BENCH_ROOT"

TASK_SETS = ("prior_auth", "appeals_denials", "dme")
DIFFICULTIES = ("easy", "medium", "hard")
PROMPT_MODES = ("zero_shot", "general", "task_specific")
OBSERVATION_MODES = ("screenshot_only", "axtree_only", "both")
ACTION_SPACES = ("dom", "coordinate")
DEFAULT_ENV_BASE_URL = "https://emrportal.vercel.app"
DEFAULT_TASK_VERSION = "v2"
