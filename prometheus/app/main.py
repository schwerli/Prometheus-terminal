import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from prometheus.app import shared_state
from prometheus.app.api import chat, repository

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  handlers=[logging.StreamHandler()],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
  # Startup event: Initialize shared_state
  app.state.shared_state = shared_state.SharedState()
  yield
  # Shutdown event: Close shared_state
  app.state.shared_state.close()


app = FastAPI(lifespan=lifespan)

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(repository.router, prefix="/repository", tags=["repository"])
