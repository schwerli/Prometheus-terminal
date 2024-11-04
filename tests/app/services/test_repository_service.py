from pathlib import Path
from unittest.mock import Mock, create_autospec, patch

import pytest

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph


@pytest.fixture
def mock_kg_service():
  service = create_autospec(KnowledgeGraphService, instance=True)
  service.kg = None
  service.kg_handler = Mock(name="kg_handler")
  return service


@pytest.fixture
def mock_git_repository():
  return create_autospec(GitRepository, instance=True)


@pytest.fixture
def mock_knowledge_graph():
  kg = create_autospec(KnowledgeGraph, instance=True)
  kg.is_built_from_github.return_value = False
  return kg


@pytest.fixture
def service(mock_kg_service, mock_git_repository):
  working_dir = Path("/test/working/dir")
  # Patch both GitRepository and Path
  with patch(
    "prometheus.app.services.repository_service.GitRepository", return_value=mock_git_repository
  ):
    return RepositoryService(
      kg_service=mock_kg_service,
      github_token="dummy_token",
      max_ast_depth=5,
      working_dir=working_dir,
    )


def test_upload_local_repository(service, mock_kg_service, mock_knowledge_graph):
  # Setup
  test_path = Path("/test/path")
  mock_kg_service.kg_handler.write_knowledge_graph.return_value = None

  with patch(
    "prometheus.app.services.repository_service.KnowledgeGraph", return_value=mock_knowledge_graph
  ) as mock_kg_class:
    # Exercise
    service.upload_local_repository(test_path)

    # Verify
    mock_kg_service.clear.assert_called_once()
    mock_kg_class.assert_called_once_with(service.max_ast_depth)
    mock_knowledge_graph.build_graph.assert_called_once_with(test_path)
    assert mock_kg_service.kg == mock_knowledge_graph
    mock_kg_service.kg_handler.write_knowledge_graph.assert_called_once_with(mock_knowledge_graph)


def test_upload_github_repo_new_repository(
  service, mock_kg_service, mock_git_repository, mock_knowledge_graph
):
  # Setup
  test_url = "https://github.com/test/repo"
  test_commit = "abc123"
  saved_path = Path("/test/saved")
  target_directory = service.working_dir / "repositories"

  mock_git_repository.has_repository.return_value = False
  mock_git_repository.clone_repository.return_value = saved_path

  with (
    patch(
      "prometheus.app.services.repository_service.KnowledgeGraph", return_value=mock_knowledge_graph
    ) as mock_kg_class,
    patch("pathlib.Path.mkdir") as mock_mkdir,
  ):  # Add mock for mkdir
    # Exercise
    service.upload_github_repo(test_url, test_commit)

    # Verify
    mock_kg_service.clear.assert_called_once()
    mock_git_repository.has_repository.assert_called_once()
    mock_git_repository.remove_repository.assert_not_called()
    mock_git_repository.clone_repository.assert_called_once_with(test_url, target_directory)
    mock_git_repository.checkout_commit.assert_called_once_with(test_commit)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    mock_kg_class.assert_called_once_with(service.max_ast_depth)
    mock_knowledge_graph.build_graph.assert_called_once_with(saved_path)
    assert mock_kg_service.kg == mock_knowledge_graph
    mock_kg_service.kg_handler.write_knowledge_graph.assert_called_once_with(mock_knowledge_graph)


def test_upload_github_repo_skips_when_should_skip(
  service, mock_kg_service, mock_git_repository, mock_knowledge_graph
):
  # Setup
  test_url = "https://github.com/test/repo"
  test_commit = "abc123"
  mock_kg = create_autospec(KnowledgeGraph, instance=True)
  mock_kg.is_built_from_github.return_value = True
  mock_kg.get_codebase_https_url.return_value = test_url
  mock_kg.get_codebase_commit_id.return_value = test_commit
  mock_kg_service.kg = mock_kg

  # Exercise
  service.upload_github_repo(test_url, test_commit)

  # Verify no actions were taken
  mock_kg_service.clear.assert_not_called()
  mock_git_repository.has_repository.assert_not_called()
  mock_git_repository.clone_repository.assert_not_called()
