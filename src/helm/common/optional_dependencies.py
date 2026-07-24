from typing import List, Optional


class OptionalDependencyNotInstalled(Exception):
    pass


# Default install hints for MedHELM when callers do not pass a specific extra.
# Prefer documented tiers over medhelm[all], which pulls heavy HEIM/VLM deps.
_DEFAULT_SUGGESTIONS = ["medhelm", "summarization", "gated"]


def handle_module_not_found_error(e: ModuleNotFoundError, suggestions: Optional[List[str]] = None):
    # Prefer caller-provided extras (e.g. medhelm[ai21]); otherwise suggest documented tiers.
    extras = suggestions if suggestions else _DEFAULT_SUGGESTIONS
    suggested_commands = " or ".join([f'`pip install "medhelm[{suggestion}]"`' for suggestion in extras])
    raise OptionalDependencyNotInstalled(
        f"Optional dependency {e.name} is not installed. Please run {suggested_commands} to install it."
    ) from e
