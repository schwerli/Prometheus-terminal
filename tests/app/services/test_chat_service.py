from unittest.mock import Mock, create_autospec, patch

import pytest

from prometheus.app.services.chat_service import ChatService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.context_provider_subgraph import ContextProviderSubgraph


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
def mock_cp_subgraph():
  return create_autospec(ContextProviderSubgraph, instance=True)


@pytest.fixture
def chat_service(
  mock_kg_service, mock_neo4j_service, mock_postgres_service, mock_llm_service, mock_cp_subgraph
):
  with patch(
    "prometheus.lang_graph.subgraphs.context_provider_subgraph.ContextProviderSubgraph",
    return_value=mock_cp_subgraph,
  ):
    service = ChatService(
      kg_service=mock_kg_service,
      neo4j_service=mock_neo4j_service,
      postgres_service=mock_postgres_service,
      llm_serivice=mock_llm_service,
    )
    return service


def test_initialization_with_kg(
  mock_kg_service, mock_neo4j_service, mock_postgres_service, mock_llm_service, mock_cp_subgraph
):
  # Setup
  with patch(
    "prometheus.lang_graph.subgraphs.context_provider_subgraph.ContextProviderSubgraph",
    return_value=mock_cp_subgraph,
  ) as mock_cp_class:
    # Exercise
    service = ChatService(
      kg_service=mock_kg_service,
      neo4j_service=mock_neo4j_service,
      postgres_service=mock_postgres_service,
      llm_serivice=mock_llm_service,
    )

    # Verify
    mock_cp_class.assert_called_once_with(
      mock_llm_service.model,
      mock_kg_service.kg,
      mock_neo4j_service.neo4j_driver,
      mock_postgres_service.checkpointer,
    )
    assert service.cp_subgraph == mock_cp_subgraph


def test_initialization_without_kg(
  mock_kg_service, mock_neo4j_service, mock_postgres_service, mock_llm_service, mock_cp_subgraph
):
  # Setup
  mock_kg_service.kg = None

  with patch(
    "prometheus.lang_graph.subgraphs.context_provider_subgraph.ContextProviderSubgraph",
    return_value=mock_cp_subgraph,
  ) as mock_cp_class:
    # Exercise
    service = ChatService(
      kg_service=mock_kg_service,
      neo4j_service=mock_neo4j_service,
      postgres_service=mock_postgres_service,
      llm_serivice=mock_llm_service,
    )

    # Verify
    mock_cp_class.assert_not_called()
    assert service.cp_subgraph is None


def test_chat_with_new_thread(chat_service, mock_cp_subgraph):
  # Setup
  query = "test query"
  expected_response = "test response"
  mock_cp_subgraph.invoke.return_value = expected_response

  # Exercise
  with patch("uuid.uuid4", return_value="mocked-uuid"):
    thread_id, response = chat_service.chat(query)

  # Verify
  assert thread_id == "mocked-uuid"
  assert response == expected_response
  mock_cp_subgraph.invoke.assert_called_once_with(query, "mocked-uuid")


def test_chat_without_initialized_kg(chat_service, mock_cp_subgraph):
  # Setup
  chat_service.cp_subgraph = None
  query = "test query"

  # Exercise & Verify
  with pytest.raises(ValueError, match="Knowledge graph not initialized"):
    chat_service.chat(query)
