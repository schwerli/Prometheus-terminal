from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from prometheus.configuration import config
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j import knowledge_graph_handler

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

  kg = KnowledgeGraph(config.config["knowledge_graph"]["max_ast_depth"])
  kg.build_graph(Path(local_repository.path))
  kg_handler = knowledge_graph_handler.KnowledgeGraphHandler(
    config.config["neo4j"]["uri"],
    config.config["neo4j"]["username"],
    config.config["neo4j"]["password"],
    config.config["neo4j"]["database"],
    config.config["neo4j"]["batch_size"],
  )
  kg_handler.write_knowledge_graph(kg)
  kg_handler.close()
  request.app.state.kg = kg
