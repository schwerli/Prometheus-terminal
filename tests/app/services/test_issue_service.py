from unittest.mock import Mock, create_autospec, patch

import pytest

from prometheus.app.services.issue_service import IssueService
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
def mock_ia_subgraph():
  return create_autospec(IssueAnswerSubgraph, instance=True)


@pytest.fixture
def issue_service(
  mock_kg_service, mock_neo4j_service, mock_postgres_service, mock_llm_service, mock_ia_subgraph
):
  with patch(
    "prometheus.lang_graph.subgraphs.issue_answer_subgraph.IssueAnswerSubgraph",
    return_value=mock_ia_subgraph,
  ):
    service = IssueService(
      kg_service=mock_kg_service,
      neo4j_service=mock_neo4j_service,
      postgres_service=mock_postgres_service,
      llm_serivice=mock_llm_service,
    )
    return service


def test_initialization_with_kg(
  mock_kg_service, mock_neo4j_service, mock_postgres_service, mock_llm_service, mock_ia_subgraph
):
  # Setup
  with patch(
    "prometheus.lang_graph.subgraphs.issue_answer_subgraph.IssueAnswerSubgraph",
    return_value=mock_ia_subgraph,
  ) as mock_ia_class:
    # Exercise
    service = IssueService(
      kg_service=mock_kg_service,
      neo4j_service=mock_neo4j_service,
      postgres_service=mock_postgres_service,
      llm_serivice=mock_llm_service,
    )

    # Verify
    mock_ia_class.assert_called_once_with(
      mock_llm_service.model,
      mock_kg_service.kg,
      mock_neo4j_service.neo4j_driver,
      mock_postgres_service.checkpointer,
    )
    assert service.ia_subgraph == mock_ia_subgraph


def test_initialization_without_kg(
  mock_kg_service, mock_neo4j_service, mock_postgres_service, mock_llm_service, mock_ia_subgraph
):
  # Setup
  mock_kg_service.kg = None

  with patch(
    "prometheus.lang_graph.subgraphs.issue_answer_subgraph.IssueAnswerSubgraph",
    return_value=mock_ia_subgraph,
  ) as mock_ia_class:
    # Exercise
    service = IssueService(
      kg_service=mock_kg_service,
      neo4j_service=mock_neo4j_service,
      postgres_service=mock_postgres_service,
      llm_serivice=mock_llm_service,
    )

    # Verify
    mock_ia_class.assert_not_called()
    assert service.ia_subgraph is None


def test_answer_issue_with_existing_thread(issue_service, mock_ia_subgraph):
  # Setup
  title = "Test Issue"
  body = "Test body"
  comments = [{"author": "user1", "content": "comment1"}]
  thread_id = "existing-thread-id"
  expected_response = "test response"
  mock_ia_subgraph.invoke.return_value = expected_response

  # Exercise
  response = issue_service.answer_issue(title, body, comments, thread_id)

  # Verify
  assert response == expected_response
  mock_ia_subgraph.invoke.assert_called_once_with(title, body, comments, thread_id)


def test_answer_issue_with_new_thread(issue_service, mock_ia_subgraph):
  # Setup
  title = "Test Issue"
  body = "Test body"
  comments = [{"author": "user1", "content": "comment1"}]
  expected_response = "test response"
  mock_ia_subgraph.invoke.return_value = expected_response

  # Exercise
  with patch("uuid.uuid4", return_value="mocked-uuid"):
    response = issue_service.answer_issue(title, body, comments)

  # Verify
  assert response == expected_response
  mock_ia_subgraph.invoke.assert_called_once_with(title, body, comments, "mocked-uuid")


def test_answer_issue_without_initialized_kg(issue_service):
  # Setup
  issue_service.ia_subgraph = None
  title = "Test Issue"
  body = "Test body"
  comments = []

  # Exercise & Verify
  with pytest.raises(ValueError, match="Knowledge graph not initialized"):
    issue_service.answer_issue(title, body, comments)
