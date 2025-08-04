from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph


@pytest.fixture
def mock_kg_service():
    service = create_autospec(KnowledgeGraphService, instance=True)
    service.get_local_path.return_value = None
    return service


@pytest.fixture
def mock_git_repository():
    repo = create_autospec(GitRepository, instance=True)
    repo.get_working_directory.return_value = Path("/test/working/dir/repositories/repo")
    return repo


@pytest.fixture
def mock_knowledge_graph():
    kg = create_autospec(KnowledgeGraph, instance=True)
    kg.is_built_from_github.return_value = False
    return kg


@pytest.fixture
def service(mock_kg_service):
    working_dir = "/test/working/dir"
    # Don't mock GitRepository in the fixture anymore
    with patch("pathlib.Path.mkdir"):
        return RepositoryService(
            kg_service=mock_kg_service,
            working_dir=working_dir,
        )


def test_clone_new_github_repo(service, mock_kg_service, mock_git_repository):
    # Setup
    test_url = "https://github.com/test/repo"
    test_commit = "abc123"
    test_github_token = "test_token"
    expected_path = Path("/test/working/dir/repositories/repo")
    mock_kg_service.local_path = None

    # Mock the GitRepository class creation
    with patch(
        "prometheus.app.services.repository_service.GitRepository", return_value=mock_git_repository
    ) as mock_git_class:
        # Exercise
        result_path = service.clone_github_repo(test_github_token, test_url, test_commit)

        # Verify
        # Check that GitRepository was instantiated with correct parameters
        mock_git_class.assert_called_once_with(
            test_url, service.target_directory, github_access_token=test_github_token
        )

        # Verify checkout_commit was called
        mock_git_repository.checkout_commit.assert_called_once_with(test_commit)

        # Verify the returned path matches expected
        assert result_path == expected_path


def test_clean_working_directory(service):
    # Setup
    with patch("shutil.rmtree") as mock_rmtree, patch("pathlib.Path.mkdir") as mock_mkdir:
        # Exercise
        service.clean()

        # Verify
        mock_rmtree.assert_called_once_with(service.target_directory)
        mock_mkdir.assert_called_once_with(parents=True)
