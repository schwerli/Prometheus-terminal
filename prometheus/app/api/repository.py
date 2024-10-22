from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class LocalRepository(BaseModel):
  path: str


@router.post(
  "/local/",
  responses={
    404: {"description": "Local repository not found"},
    200: {"description": "Repository uploaded successfully"},
  },
)
def upload_local_repository(local_repository: LocalRepository, request: Request):
  local_path = Path(local_repository.path)

  if not local_path.exists():
    raise HTTPException(
      status_code=404,
      detail=f"Local repository not found at path: {local_repository.path}",
    )

  request.app.state.shared_state.upload_repository(local_path)

  return {"message": "Repository uploaded successfully"}


@router.get("/delete/")
def delete(request: Request):
  if not request.app.state.shared_state.kg_handler.knowledge_graph_exists():
    return {"message": "No knowledge graph to delete"}

  request.app.state.shared_state.clear_knowledge_graph()
  return {"message": "Successfully deleted knowledge graph"}


@router.get("/exists/", response_model=bool)
def knowledge_graph_exists(request: Request) -> bool:
  return request.app.state.shared_state.kg_handler.knowledge_graph_exists()
