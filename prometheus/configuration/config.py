from pathlib import Path

from dynaconf import Dynaconf

settings_file = Path(__file__).resolve().parent / "settings.toml"

settings = Dynaconf(
  envvar_prefix="PROMETHEUS",
  settings_files=[str(settings_file)],
  environments=True,
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
