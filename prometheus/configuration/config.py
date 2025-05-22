from dynaconf import Dynaconf

settings = Dynaconf(envvar_prefix="PROMETHEUS", load_dotenv=True)
