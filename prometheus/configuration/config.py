import json
import os
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.json"


def load_config():
  if not CONFIG_FILE.exists():
    return {}

  with open(CONFIG_FILE, "r") as f:
    config_data = json.load(f)

  if "litellm" in config_data:
    if "anthropic_api_key" in config_data["litellm"]:
      os.environ["ANTHROPIC_API_KEY"] = config_data["litellm"]["anthropic_api_key"]
    elif "azure_api_key" in config_data["litellm"]:
      os.environ["AZURE_API_KEY"] = config_data["litellm"]["azure_api_key"]
    elif "openai_api_key" in config_data["litellm"]:
      os.environ["OPENAI_API_KEY"] = config_data["litellm"]["openai_api_key"]
  return config_data


config = load_config()
