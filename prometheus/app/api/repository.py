import git
from fastapi import APIRouter, Request

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.exceptions.server_exception import ServerException

router = APIRouter()


@router.get(
    "/github/",
    description="""
    Upload a GitHub repository to Prometheus, default to the latest commit in the main branch.
    """,
)
def upload_github_repository(github_token: str, https_url: str, request: Request):
    # Get the repository and knowledge graph services
    repository_service: RepositoryService = request.app.state.service["repository_service"]
    knowledge_graph_service: KnowledgeGraphService = request.app.state.service[
        "knowledge_graph_service"
    ]

    # Clean the services to ensure no previous data is present
    repository_service.clean()
    knowledge_graph_service.clear()
    try:
        # Clone the repository
        saved_path = repository_service.clone_github_repo(github_token, https_url)
    except git.exc.GitCommandError:
        raise ServerException(code=400, message=f"Unable to clone {https_url}")
    # Build and save the knowledge graph from the cloned repository
    knowledge_graph_service.build_and_save_knowledge_graph(saved_path, https_url)


@router.get(
    "/github_commit/",
    description="""
    Upload a GitHub repository at a specific commit to Prometheus.
    """,
)
def upload_github_repository_at_commit(
    github_token, https_url: str, commit_id: str, request: Request
):
    # Get the repository and knowledge graph services
    repository_service: RepositoryService = request.app.state.service["repository_service"]
    knowledge_graph_service: KnowledgeGraphService = request.app.state.service[
        "knowledge_graph_service"
    ]

    # Clean the services to ensure no previous data is present
    repository_service.clean()
    knowledge_graph_service.clear()

    try:
        # Clone the repository
        saved_path = repository_service.clone_github_repo(github_token, https_url, commit_id)
    except git.exc.GitCommandError:
        raise ServerException(code=400, message=f"Unable to clone {https_url}")
    # Build and save the knowledge graph from the cloned repository
    knowledge_graph_service.build_and_save_knowledge_graph(saved_path, https_url, commit_id)


@router.get(
    "/delete/",
    description="""
    Delete the repository uploaded to Prometheus, along with other information.
    """,
)
def delete(request: Request):
    knowledge_graph_service: KnowledgeGraphService = request.app.state.service[
        "knowledge_graph_service"
    ]
    if not knowledge_graph_service.exists():
        return {"message": "No knowledge graph to delete"}
    # Get the repository service to clean up the repository data
    repository_service: RepositoryService = request.app.state.service["repository_service"]

    # Clear the knowledge graph and repository data
    knowledge_graph_service.clear()
    repository_service.clean()
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
