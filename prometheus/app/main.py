import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from prometheus.app import dependencies
from prometheus.app.api import issue, repository
from prometheus.configuration.config import settings

# Create a logger for application's namespace
logger = logging.getLogger("prometheus")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(getattr(logging, settings.LOGGING_LEVEL))
logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
  app.state.service_coordinator = dependencies.initialize_services()
  yield
  app.state.service_coordiantor.clear()
  app.state.service_coordinator.close()


app = FastAPI(lifespan=lifespan)

app.include_router(repository.router, prefix="/repository", tags=["repository"])
app.include_router(issue.router, prefix="/issue", tags=["issue"])
