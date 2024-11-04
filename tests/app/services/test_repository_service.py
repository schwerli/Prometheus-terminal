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


def test_clone_new_github_repo(service, mock_kg_service, mock_git_repository):
  # Setup
  test_url = "https://github.com/test/repo"
  test_commit = "abc123"
  saved_path = Path("/test/saved")
  target_directory = service.target_directory

  mock_git_repository.has_repository.return_value = False
  mock_git_repository.clone_repository.return_value = saved_path

  # Exercise
  result_path = service.clone_github_repo(test_url, test_commit)

  # Verify
  mock_git_repository.has_repository.assert_called_once()
  mock_git_repository.remove_repository.assert_not_called()
  mock_git_repository.clone_repository.assert_called_once_with(test_url, target_directory)
  mock_git_repository.checkout_commit.assert_called_once_with(test_commit)
  assert result_path == saved_path


def test_clone_github_repo_with_existing_repo(service, mock_kg_service, mock_git_repository):
  # Setup
  test_url = "https://github.com/test/repo"
  test_commit = "abc123"
  saved_path = Path("/test/saved")
  target_directory = service.target_directory

  mock_git_repository.has_repository.return_value = True
  mock_git_repository.clone_repository.return_value = saved_path

  # Exercise
  result_path = service.clone_github_repo(test_url, test_commit)

  # Verify
  mock_git_repository.has_repository.assert_called_once()
  mock_git_repository.remove_repository.assert_called_once()
  mock_git_repository.clone_repository.assert_called_once_with(test_url, target_directory)
  mock_git_repository.checkout_commit.assert_called_once_with(test_commit)
  assert result_path == saved_path


def test_skip_clone_when_already_loaded(service, mock_kg_service, mock_git_repository):
  # Setup
  test_url = "https://github.com/test/repo"
  test_commit = "abc123"

  mock_knowledge_graph = Mock()
  mock_knowledge_graph.is_built_from_github.return_value = True
  mock_knowledge_graph.get_codebase_https_url.return_value = test_url
  mock_knowledge_graph.get_codebase_commit_id.return_value = test_commit

  mock_kg_service.kg = mock_knowledge_graph

  # Exercise
  result_path = service.clone_github_repo(test_url, test_commit)

  # Verify
  assert result_path is None
  mock_git_repository.has_repository.assert_not_called()
  mock_git_repository.clone_repository.assert_not_called()
  mock_git_repository.checkout_commit.assert_not_called()


def test_clean_working_directory(service):
  # Setup
  with patch("shutil.rmtree") as mock_rmtree, patch("pathlib.Path.mkdir") as mock_mkdir:
    # Exercise
    service.clean_working_directory()

    # Verify
    mock_rmtree.assert_called_once_with(service.target_directory)
    mock_mkdir.assert_called_once_with(parents=True)
