from unittest.mock import Mock, create_autospec, patch

import pytest

from prometheus.app.services.issue_answer_service import IssueAnswerService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.issue_answer_subgraph import IssueAnswerSubgraph


@pytest.fixture
def mock_kg():
  return create_autospec(KnowledgeGraph, instance=True)


@pytest.fixture
def mock_kg_service(mock_kg):
  service = create_autospec(KnowledgeGraphService, instance=True)
  service.kg = mock_kg
  return service


@pytest.fixture
def mock_neo4j_service():
  service = create_autospec(Neo4jService, instance=True)
  service.neo4j_driver = Mock(name="neo4j_driver")
  return service


@pytest.fixture
def mock_postgres_service():
  service = create_autospec(PostgresService, instance=True)
  service.checkpointer = Mock(name="checkpointer")
  return service


@pytest.fixture
def mock_llm_service():
  service = create_autospec(LLMService, instance=True)
  service.model = Mock(name="llm_model")
  return service


@pytest.fixture
def mock_issue_answer_subgraph():
  return create_autospec(IssueAnswerSubgraph, instance=True)


@pytest.fixture
def issue_service(
  mock_kg_service,
  mock_neo4j_service,
  mock_postgres_service,
  mock_llm_service,
  mock_issue_answer_subgraph,
):
  with patch(
    "prometheus.app.services.issue_answer_service.IssueAnswerSubgraph",
    return_value=mock_issue_answer_subgraph,
  ):
    service = IssueAnswerService(
      kg_service=mock_kg_service,
      neo4j_service=mock_neo4j_service,
      postgres_service=mock_postgres_service,
      llm_serivice=mock_llm_service,
    )
    return service


def test_initialization_with_kg(
  mock_kg_service,
  mock_neo4j_service,
  mock_postgres_service,
  mock_llm_service,
  mock_issue_answer_subgraph,
):
  # Setup
  with patch(
    "prometheus.app.services.issue_answer_service.IssueAnswerSubgraph",
    return_value=mock_issue_answer_subgraph,
  ) as mock_ia_class:
    # Exercise
    service = IssueAnswerService(
      kg_service=mock_kg_service,
      neo4j_service=mock_neo4j_service,
      postgres_service=mock_postgres_service,
      llm_serivice=mock_llm_service,
    )

    # Verify
    mock_ia_class.assert_called_once_with(
      service.model,  # Updated to use service.model instead of mock_llm_service.model
      mock_kg_service.kg,
      mock_neo4j_service.neo4j_driver,
      mock_postgres_service.checkpointer,
    )
    assert service.issue_answer_subgraph == mock_issue_answer_subgraph
    assert service.model == mock_llm_service.model  # Add verification for model assignment


def test_initialization_without_kg(
  mock_kg_service,
  mock_neo4j_service,
  mock_postgres_service,
  mock_llm_service,
  mock_issue_answer_subgraph,
):
  # Setup
  mock_kg_service.kg = None

  with patch(
    "prometheus.lang_graph.subgraphs.issue_answer_subgraph.IssueAnswerSubgraph",
    return_value=mock_issue_answer_subgraph,
  ) as mock_ia_class:
    # Exercise
    service = IssueAnswerService(
      kg_service=mock_kg_service,
      neo4j_service=mock_neo4j_service,
      postgres_service=mock_postgres_service,
      llm_serivice=mock_llm_service,
    )

    # Verify
    mock_ia_class.assert_not_called()
    assert service.issue_answer_subgraph is None
    assert service.model == mock_llm_service.model  # Add verification for model assignment


def test_answer_issue_with_existing_thread(issue_service, mock_issue_answer_subgraph):
  # Setup
  title = "Test Issue"
  body = "Test body"
  comments = [{"author": "user1", "content": "comment1"}]
  thread_id = "existing-thread-id"
  expected_response = "test response"
  mock_issue_answer_subgraph.invoke.return_value = expected_response

  # Exercise
  response = issue_service.answer_issue(title, body, comments, thread_id)

  # Verify
  assert response == expected_response
  mock_issue_answer_subgraph.invoke.assert_called_once_with(title, body, comments, thread_id)


def test_answer_issue_with_new_thread(issue_service, mock_issue_answer_subgraph):
  # Setup
  title = "Test Issue"
  body = "Test body"
  comments = [{"author": "user1", "content": "comment1"}]
  expected_response = "test response"
  mock_issue_answer_subgraph.invoke.return_value = expected_response

  # Exercise
  with patch("uuid.uuid4", return_value="mocked-uuid"):
    response = issue_service.answer_issue(title, body, comments)

  # Verify
  assert response == expected_response
  mock_issue_answer_subgraph.invoke.assert_called_once_with(title, body, comments, "mocked-uuid")


def test_answer_issue_without_initialized_kg(issue_service):
  # Setup
  issue_service.issue_answer_subgraph = None
  title = "Test Issue"
  body = "Test body"
  comments = []

  # Exercise & Verify
  with pytest.raises(ValueError, match="Knowledge graph not initialized"):
    issue_service.answer_issue(title, body, comments)
