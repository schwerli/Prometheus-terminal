from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from prometheus.app.services.issue_answer_and_fix_service import IssueAnswerAndFixService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.app.services.service_coordinator import ServiceCoordinator


@pytest.fixture
def mock_services():
  llm_service = Mock(spec=LLMService)
  llm_service.model = Mock()

  knowledge_graph_service = Mock(spec=KnowledgeGraphService)
  knowledge_graph_service.kg = None
  knowledge_graph_service.local_path = None
  neo4j_service = Mock(spec=Neo4jService)
  postgres_service = Mock(spec=PostgresService)
  repository_service = Mock(spec=RepositoryService)

  return {
    "knowledge_graph_service": knowledge_graph_service,
    "llm_service": llm_service,
    "neo4j_service": neo4j_service,
    "postgres_service": postgres_service,
    "repository_service": repository_service,
    "github_token": "test_token",
    "working_directory": "test_directory",
  }


@pytest.fixture
def mock_issue_answer_and_fix_service():
  return Mock(spec=IssueAnswerAndFixService)


@pytest.fixture
def service_coordinator(mock_services, mock_issue_answer_and_fix_service):
  with (
    patch(
      "prometheus.app.services.service_coordinator.IssueAnswerAndFixService",
      return_value=mock_issue_answer_and_fix_service,
    ),
  ):
    coordinator = ServiceCoordinator(
      mock_services["knowledge_graph_service"],
      mock_services["llm_service"],
      mock_services["neo4j_service"],
      mock_services["postgres_service"],
      mock_services["repository_service"],
      mock_services["github_token"],
      mock_services["working_directory"],
    )
    return coordinator


def test_initialization(service_coordinator, mock_services):
  """Test that services are properly initialized"""
  assert service_coordinator.knowledge_graph_service == mock_services["knowledge_graph_service"]
  assert service_coordinator.llm_service == mock_services["llm_service"]
  assert service_coordinator.neo4j_service == mock_services["neo4j_service"]
  assert service_coordinator.postgres_service == mock_services["postgres_service"]
  assert service_coordinator.repository_service == mock_services["repository_service"]


def test_upload_local_repository(service_coordinator, mock_services):
  """Test local repository upload process"""
  test_path = Path("/test/path")

  service_coordinator.upload_local_repository(test_path)

  # Verify the sequence of operations
  mock_services["knowledge_graph_service"].clear.assert_called_once()
  mock_services["knowledge_graph_service"].build_and_save_knowledge_graph.assert_called_once_with(
    test_path
  )


def test_upload_github_repository(service_coordinator, mock_services):
  """Test GitHub repository upload process"""
  test_url = "https://github.com/test/repo"
  test_commit = "abc123"
  saved_path = Path("/saved/path")
  mock_services["repository_service"].clone_github_repo.return_value = saved_path

  service_coordinator.upload_github_repository(test_url, test_commit)

  # Verify the sequence of operations
  mock_services["knowledge_graph_service"].clear.assert_called_once()
  mock_services["repository_service"].clone_github_repo.assert_called_once_with(
    mock_services["github_token"], test_url, test_commit
  )
  mock_services["knowledge_graph_service"].build_and_save_knowledge_graph.assert_called_once_with(
    saved_path, test_url, test_commit
  )


def test_get_all_conversation_ids(service_coordinator, mock_services):
  """Test retrieval of conversation IDs"""
  expected_ids = ["thread1", "thread2"]
  mock_services["postgres_service"].get_all_thread_ids.return_value = expected_ids

  result = service_coordinator.get_all_conversation_ids()

  mock_services["postgres_service"].get_all_thread_ids.assert_called_once()
  assert result == expected_ids


def test_get_messages(service_coordinator, mock_services):
  """Test message retrieval for a conversation"""
  expected_messages = [{"role": "user", "content": "message1"}]
  mock_services["postgres_service"].get_messages.return_value = expected_messages

  result = service_coordinator.get_messages("thread123")

  mock_services["postgres_service"].get_messages.assert_called_once_with("thread123")
  assert result == expected_messages


def test_clear(service_coordinator, mock_services):
  """Test clearing of services"""
  service_coordinator.clear()

  mock_services["knowledge_graph_service"].clear.assert_called_once()
  mock_services["repository_service"].clean_working_directory.assert_called_once()


def test_close(service_coordinator, mock_services):
  """Test proper closing of services"""
  service_coordinator.close()

  mock_services["neo4j_service"].close.assert_called_once()
  mock_services["postgres_service"].close.assert_called_once()


def test_exists_knowledge_graph(service_coordinator, mock_services):
  """Test knowledge graph existence check"""
  mock_services["knowledge_graph_service"].exists.return_value = True

  result = service_coordinator.exists_knowledge_graph()

  mock_services["knowledge_graph_service"].exists.assert_called_once()
  assert result is True
