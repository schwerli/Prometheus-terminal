import os
from pathlib import Path

from dynaconf import Dynaconf

settings_file = Path(__file__).resolve().parent / "settings.toml"

settings = Dynaconf(
  envvar_prefix="PROMETHEUS",
  settings_files=[str(settings_file)],
  environments=True,
)

if "LITELLM_ANTHROPIC_API_KEY" in settings:
  os.environ["ANTHROPIC_API_KEY"] = settings.LITELLM_ANTHROPIC_API_KEY
if "LITELLM_GEMINI_API_KEY" in settings:
  os.environ["GEMINI_API_KEY"] = settings.LITELLM_GEMINI_API_KEY
