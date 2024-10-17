import logging
from fastapi import FastAPI

from prometheus.app.api import chat, repository

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  handlers=[logging.StreamHandler()],
)

app = FastAPI()

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(repository.router, prefix="/repository", tags=["repository"])
