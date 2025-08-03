import git
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get(
    "/github/",
    description="""
    Upload a GitHub repository to Prometheus, default to the latest commit in the main branch.
    """,
)
def upload_github_repository(github_token: str, https_url: str, request: Request):
    try:
        # Get the repository and knowledge graph services
        repository_service = request.app.state.service["repository_service"]
        knowledge_graph_service = request.app.state.service["knowledge_graph_service"]

        # Clean the services to ensure no previous data is present
        repository_service.clean()
        knowledge_graph_service.clean()

        # Clone the repository
        saved_path = repository_service.clone_github_repo(github_token, https_url)
        # Build and save the knowledge graph from the cloned repository
        knowledge_graph_service.build_and_save_knowledge_graph(saved_path, https_url)
    except git.exc.GitCommandError:
        raise HTTPException(status_code=400, detail=f"Unable to clone {https_url}")


@router.get(
    "/github_commit/",
    description="""
    Upload a GitHub repository at a specific commit to Prometheus.
    """,
)
def upload_github_repository_at_commit(
    github_token, https_url: str, commit_id: str, request: Request
):
    try:
        # Get the repository and knowledge graph services
        repository_service = request.app.state.service["repository_service"]
        knowledge_graph_service = request.app.state.service["knowledge_graph_service"]

        # Clean the services to ensure no previous data is present
        repository_service.clean()
        knowledge_graph_service.clean()

        # Clone the repository
        saved_path = repository_service.clone_github_repo(github_token, https_url, commit_id)

        # Build and save the knowledge graph from the cloned repository
        knowledge_graph_service.build_and_save_knowledge_graph(saved_path, https_url, commit_id)
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
    if not request.app.state.service["knowledge_graph_service"].exists():
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
    return request.app.state.service["knowledge_graph_service"].exists()
