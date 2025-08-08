from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest

from prometheus.app.services.database_service import DatabaseService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.git.git_repository import GitRepository


@pytest.fixture
def mock_kg_service():
    # Mock KnowledgeGraphService; RepositoryService only reads its attributes in other paths
    return create_autospec(KnowledgeGraphService, instance=True)


@pytest.fixture
def mock_database_service():
    # Mock DatabaseService with an engine attribute (used by RepositoryService.__init__)
    ds = create_autospec(DatabaseService, instance=True)
    ds.engine = create_autospec(object, instance=True)
    return ds


@pytest.fixture
def mock_git_repository():
    repo = create_autospec(GitRepository, instance=True)
    repo.get_working_directory.return_value = Path("/test/working/dir/repositories/repo")
    return repo


@pytest.fixture
def service(mock_kg_service, mock_database_service, monkeypatch):
    working_dir = "/test/working/dir"
    # Avoid touching the real filesystem when creating the base repo folder
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    return RepositoryService(
        kg_service=mock_kg_service,
        database_service=mock_database_service,  # <-- use the correct fixture here
        working_dir=working_dir,
    )


def test_clone_new_github_repo(service, mock_git_repository, monkeypatch):
    # Arrange
    test_url = "https://github.com/test/repo"
    test_commit = "abc123"
    test_github_token = "test_token"
    expected_path = Path("/test/working/dir/repositories/repo")

    # Force get_new_playground_path() to return a deterministic path for assertions
    monkeypatch.setattr(service, "get_new_playground_path", lambda: expected_path)

    # Patch GitRepository so its constructor returns our mock instance
    with patch(
        "prometheus.app.services.repository_service.GitRepository",
        return_value=mock_git_repository,
    ) as mock_git_class:
        # Act
        result_path = service.clone_github_repo(test_github_token, test_url, test_commit)

        # Assert

        # GitRepository should be instantiated without args (per current implementation)
        mock_git_class.assert_called_once_with()

        # Ensure the clone method was invoked with correct parameters
        mock_git_repository.from_clone_repository.assert_called_once_with(
            test_url, test_github_token, expected_path
        )

        # Verify the requested commit was checked out
        mock_git_repository.checkout_commit.assert_called_once_with(test_commit)

        # The returned path should be the working directory of the mocked repo
        assert result_path == expected_path
