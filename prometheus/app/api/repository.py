from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from prometheus.configuration import config
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j import handler

router = APIRouter()


class LocalRepository(BaseModel):
  path: str


@router.post("/local/", responses={404: {"description": "Local repository not found"}})
def upload_local_repository(local_repository: LocalRepository, request: Request):
  local_path = Path(local_repository.path)

  if not local_path.exists():
    raise HTTPException(
      status_code=404,
      detail=f"Local repository not found at path: {local_repository.path}",
    )

  kg = KnowledgeGraph(
    Path(local_repository.path), config.config["knowledge_graph"]["max_ast_depth"]
  )
  neo4j_handler = handler.Handler(
    config.config["neo4j"]["uri"],
    config.config["neo4j"]["username"],
    config.config["neo4j"]["password"],
    config.config["neo4j"]["database"],
    config.config["neo4j"]["batch_size"],
  )
  neo4j_handler.write_knowledge_graph(kg)
  neo4j_handler.close()
  request.app.state.kg = kg
