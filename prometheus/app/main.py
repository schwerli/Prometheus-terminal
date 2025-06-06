import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from prometheus.app import dependencies
from prometheus.app.api import issue, repository
from prometheus.configuration.config import settings

# Create a logger for the application's namespace
logger = logging.getLogger("prometheus")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(getattr(logging, settings.LOGGING_LEVEL))
logger.propagate = False

logger.info(f"LOGGING_LEVEL={settings.LOGGING_LEVEL}")
logger.info(f"ADVANCED_MODEL={settings.ADVANCED_MODEL}")
logger.info(f"BASE_MODEL={settings.BASE_MODEL}")
logger.info(f"NEO4J_BATCH_SIZE={settings.NEO4J_BATCH_SIZE}")
logger.info(f"WORKING_DIRECTORY={settings.WORKING_DIRECTORY}")
logger.info(f"KNOWLEDGE_GRAPH_MAX_AST_DEPTH={settings.KNOWLEDGE_GRAPH_MAX_AST_DEPTH}")
logger.info(f"KNOWLEDGE_GRAPH_CHUNK_SIZE={settings.KNOWLEDGE_GRAPH_CHUNK_SIZE}")
logger.info(f"KNOWLEDGE_GRAPH_CHUNK_OVERLAP={settings.KNOWLEDGE_GRAPH_CHUNK_OVERLAP}")
logger.info(f"MAX_TOKEN_PER_NEO4J_RESULT={settings.MAX_TOKEN_PER_NEO4J_RESULT}")
logger.info(f"MAX_TOKENS={settings.MAX_TOKENS}")


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
