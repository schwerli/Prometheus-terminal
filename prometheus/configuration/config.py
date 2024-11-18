from dynaconf import Dynaconf

settings = Dynaconf(
  envvar_prefix="PROMETHEUS",
  settings_files=["settings.toml"],
  environments=True,
)
