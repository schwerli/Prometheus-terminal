import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from prometheus.app import dependencies
from prometheus.app.api import issue, repository
from prometheus.configuration.config import settings

# Create a logger for application's namespace
logger = logging.getLogger("prometheus")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(getattr(logging, settings.LOGGING_LEVEL))
logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
  app.state.service_coordinator = dependencies.initialize_services()
  yield
  app.state.service_coordinator.clear()
  app.state.service_coordinator.close()


app = FastAPI(lifespan=lifespan)

app.include_router(repository.router, prefix="/repository", tags=["repository"])
app.include_router(issue.router, prefix="/issue", tags=["issue"])


@app.get("/health", tags=["health"])
def health_check():
  return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
