import logging
from contextlib import contextmanager

from fastapi import FastAPI

from prometheus.app import shared_state
from prometheus.app.api import chat, repository

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  handlers=[logging.StreamHandler()],
)

app = FastAPI()


@contextmanager
def startup_event():
  app.state.shared_state = shared_state.SharedState()
  yield
  app.state.shared_state.close()


app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(repository.router, prefix="/repository", tags=["repository"])
