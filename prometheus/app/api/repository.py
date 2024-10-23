from pathlib import Path

import git
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get(
  "/local/",
  responses={
    404: {"description": "Local repository not found"},
    200: {"description": "Repository uploaded successfully"},
  },
)
def upload_local_repository(local_repository: str, request: Request):
  local_path = Path(local_repository)

  if not local_path.exists():
    raise HTTPException(
      status_code=404,
      detail=f"Local repository not found at path: {local_repository}",
    )

  request.app.state.shared_state.upload_local_repository(local_path)

  return {"message": "Repository uploaded successfully"}


@router.get("/github/")
def upload_github_repository(https_url: str, request: Request):
  try:
    request.app.state.shared_state.upload_github_repository(https_url)
  except git.exc.GitCommandError:
    raise HTTPException(status_code=400, detail=f"Unable to clone {https_url}")


@router.get("/delete/")
def delete(request: Request):
  if not request.app.state.shared_state.kg_handler.knowledge_graph_exists():
    return {"message": "No knowledge graph to delete"}

  request.app.state.shared_state.clear_knowledge_graph()
  return {"message": "Successfully deleted knowledge graph"}


@router.get("/exists/", response_model=bool)
def knowledge_graph_exists(request: Request) -> bool:
  return request.app.state.shared_state.kg_handler.knowledge_graph_exists()
