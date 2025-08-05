import git
from fastapi import APIRouter, Request

from prometheus.app.decorators.require_login import requireLogin
from prometheus.app.models.response.response import Response
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.app.services.user_service import UserService
from prometheus.exceptions.server_exception import ServerException

router = APIRouter()


def get_github_token(request: Request, github_token: str) -> str:
    """Retrieve GitHub token from the request or user profile."""
    # If the token is provided in the request, use it directly
    if github_token:
        return github_token
    # If the token is not provided, fetch it from the user profile if logged in
    # Check if the user is authenticated
    if not request.state.user_id:
        # If the user is not authenticated, raise an exception
        raise ServerException(
            code=401, message="GitHub token is required, please provide it or log in"
        )
    # If the user is authenticated, get the user service and fetch the token
    user_service: UserService = request.app.state.service["user_service"]
    user = user_service.get_user_by_id(request.state.user_id)
    github_token = user.github_token if user else None

    # If the token is still not available, raise an exception
    if not github_token:
        raise ServerException(
            code=423, message="Either provide a GitHub token or set it in your user profile"
        )
    return github_token


@router.get(
    "/github/",
    description="""
    Upload a GitHub repository to Prometheus, default to the latest commit in the main branch.
    """,
    response_model=Response,
)
@requireLogin
def upload_github_repository(github_token: str, https_url: str, request: Request):
    # Get the repository and knowledge graph services
    repository_service: RepositoryService = request.app.state.service["repository_service"]
    knowledge_graph_service: KnowledgeGraphService = request.app.state.service[
        "knowledge_graph_service"
    ]
    github_token = get_github_token(request, github_token)

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
    return Response()


@router.get(
    "/github_commit/",
    description="""
    Upload a GitHub repository at a specific commit to Prometheus.
    """,
    response_model=Response,
)
@requireLogin
def upload_github_repository_at_commit(
    github_token, https_url: str, commit_id: str, request: Request
):
    # Get the repository and knowledge graph services
    repository_service: RepositoryService = request.app.state.service["repository_service"]
    knowledge_graph_service: KnowledgeGraphService = request.app.state.service[
        "knowledge_graph_service"
    ]

    github_token = get_github_token(request, github_token)

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
    return Response()


@router.get(
    "/delete/",
    description="""
    Delete the repository uploaded to Prometheus, along with other information.
    """,
    response_model=Response,
)
@requireLogin
def delete(request: Request):
    knowledge_graph_service: KnowledgeGraphService = request.app.state.service[
        "knowledge_graph_service"
    ]
    if not knowledge_graph_service.exists():
        return Response(message="No knowledge graph to delete")
    # Get the repository service to clean up the repository data
    repository_service: RepositoryService = request.app.state.service["repository_service"]

    # Clear the knowledge graph and repository data
    knowledge_graph_service.clear()
    repository_service.clean()
    return Response()


@requireLogin
@router.get(
    "/exists/",
    description="""
    If there is a codebase uploaded to Promtheus.
    """,
    response_model=Response[bool],
)
def knowledge_graph_exists(request: Request) -> Response[bool]:
    return Response(data=request.app.state.service["knowledge_graph_service"].exists())
