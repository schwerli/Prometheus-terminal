import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.json"


def load_config():
  if not CONFIG_FILE.exists():
    return {}

  with open(CONFIG_FILE, "r") as f:
    return json.load(f)


config = load_config()
