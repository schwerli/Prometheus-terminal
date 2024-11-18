from pathlib import Path
from unittest.mock import Mock, create_autospec

import pytest

from prometheus.app.services.issue_answer_and_fix_service import IssueAnswerAndFixService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_subgraph import IssueAnswerAndFixSubgraph


@pytest.fixture
def mock_kg_service():
  service = create_autospec(KnowledgeGraphService, instance=True)
  service.kg = Mock(name="mock_knowledge_graph")
  return service


@pytest.fixture
def mock_neo4j_service():
  service = create_autospec(Neo4jService, instance=True)
  service.neo4j_driver = Mock(name="mock_neo4j_driver")
  return service


@pytest.fixture
def mock_postgres_service():
  service = create_autospec(PostgresService, instance=True)
  service.checkpointer = Mock(name="mock_checkpointer")
  return service


@pytest.fixture
def mock_llm_service():
  service = create_autospec(LLMService, instance=True)
  service.model = Mock(name="mock_model")
  return service


@pytest.fixture
def mock_subgraph_class(monkeypatch):
  mock_subgraph = create_autospec(IssueAnswerAndFixSubgraph, instance=True)
  mock_class = Mock(return_value=mock_subgraph)
  monkeypatch.setattr(
    "prometheus.app.services.issue_answer_and_fix_service.IssueAnswerAndFixSubgraph", mock_class
  )
  return mock_class, mock_subgraph


def test_init_with_knowledge_graph(
  mock_kg_service, mock_neo4j_service, mock_postgres_service, mock_llm_service, mock_subgraph_class
):
  # Setup
  mock_class, mock_subgraph = mock_subgraph_class
  local_path = Path("/mock/path")

  # Exercise
  service = IssueAnswerAndFixService(
    mock_kg_service,
    mock_neo4j_service,
    1000,
    mock_postgres_service,
    mock_llm_service,
    local_path,
    dockerfile_content="FROM python:3.9",
    image_name=None,
    workdir="/app",
    build_commands=["pip install -r requirements.txt"],
    test_commands=["pytest"],
  )

  # Verify
  assert service.issue_answer_and_fix_subgraph == mock_subgraph
  mock_class.assert_called_once_with(
    mock_llm_service.model,
    mock_kg_service.kg,
    mock_neo4j_service.neo4j_driver,
    1000,
    local_path,
    mock_postgres_service.checkpointer,
    "FROM python:3.9",
    None,
    "/app",
    ["pip install -r requirements.txt"],
    ["pytest"],
  )


def test_init_without_knowledge_graph(
  mock_kg_service, mock_neo4j_service, mock_postgres_service, mock_llm_service, mock_subgraph_class
):
  # Setup
  mock_kg_service.kg = None
  local_path = Path("/mock/path")

  with pytest.raises(ValueError):
    IssueAnswerAndFixService(
      mock_kg_service, mock_neo4j_service, 1000, mock_postgres_service, mock_llm_service, local_path
    )


def test_answer_and_fix_issue_success(
  mock_kg_service, mock_neo4j_service, mock_postgres_service, mock_llm_service, mock_subgraph_class
):
  # Setup
  mock_kg_service.kg = Mock(name="mock_kg")
  mock_class, mock_subgraph = mock_subgraph_class
  local_path = Path("/mock/path")
  expected_response = "Mock response"
  mock_subgraph.invoke.return_value = expected_response

  # Test data
  title = "Test Issue"
  body = "Test Body"
  comments = [{"author": "user1", "comment": "test comment"}]
  thread_id = "test-thread-id"

  # Exercise
  service = IssueAnswerAndFixService(
    mock_kg_service, mock_neo4j_service, 1000, mock_postgres_service, mock_llm_service, local_path
  )
  response = service.answer_and_fix_issue(
    title,
    body,
    comments,
    response_mode="AUTO",
    run_build=False,
    run_tests=False,
    thread_id=thread_id,
  )

  # Verify
  assert response == expected_response
  mock_subgraph.invoke.assert_called_once_with(
    title, body, comments, "AUTO", False, False, thread_id
  )
