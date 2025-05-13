from pathlib import Path

import git
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get(
    "/local/",
    description="""
    Upload a local codebase to Prometheus.
    """,
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

    request.app.state.service_coordinator.upload_local_repository(local_path)

    return {"message": "Repository uploaded successfully"}


@router.get(
    "/github/",
    description="""
    Upload a GitHub repository to Prometheus, default to the latest commit in the main branch.
    """,
)
def upload_github_repository(https_url: str, request: Request):
    try:
        request.app.state.service_coordinator.upload_github_repository(https_url)
    except git.exc.GitCommandError:
        raise HTTPException(status_code=400, detail=f"Unable to clone {https_url}")


@router.get(
    "/github_commit/",
    description="""
    Upload a GitHub repository at a specific commit to Prometheus.
    """,
)
def upload_github_repository_at_commit(https_url: str, commit_id: str, request: Request):
    try:
        request.app.state.service_coordinator.upload_github_repository(https_url, commit_id)
    except git.exc.GitCommandError:
        raise HTTPException(
            status_code=400, detail=f"Unable to clone {https_url} with commit {commit_id}"
        )


@router.get(
    "/delete/",
    description="""
    Delete the repository uploaded to Prometheus, along with other information.
    """,
)
def delete(request: Request):
    if not request.app.state.service_coordinator.exists_knowledge_graph():
        return {"message": "No knowledge graph to delete"}

    request.app.state.service_coordinator.clear()
    return {"message": "Successfully deleted knowledge graph"}


@router.get(
    "/exists/",
    description="""
    If there is a codebase uploaded to Promtheus.
    """,
    response_model=bool,
)
def knowledge_graph_exists(request: Request) -> bool:
    return request.app.state.service_coordinator.exists_knowledge_graph()
