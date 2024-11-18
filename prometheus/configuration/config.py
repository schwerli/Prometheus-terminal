import os

from dynaconf import Dynaconf

settings = Dynaconf(
  envvar_prefix="PROMETHEUS",
  settings_files=["settings.toml"],
  environments=True,
)

if "LITELLM_ANTHROPIC_API_KEY" in settings:
  os.environ["ANTHROPIC_API_KEY"] = settings.LITELLM_ANTHROPIC_API_KEY
if "LITELLM_GEMINI_API_KEY" in settings:
  os.environ["GOOGLE_API_KEY"] = settings.LITELLM_GEMINI_API_KEY
if "LITELLM_OPENAI_API_KEY" in settings:
  os.environ["OPENAI_API_KEY"] = settings.LITELLM_OPENAI_API_KEY
