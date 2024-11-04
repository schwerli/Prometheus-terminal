from unittest.mock import Mock, create_autospec

import pytest

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j.knowledge_graph_handler import KnowledgeGraphHandler


@pytest.fixture
def mock_neo4j_driver():
  return Mock(name="mock_neo4j_driver")


@pytest.fixture
def mock_neo4j_service(mock_neo4j_driver):
  service = create_autospec(Neo4jService, instance=True)
  service.neo4j_driver = mock_neo4j_driver
  return service


@pytest.fixture
def mock_knowledge_graph():
  return create_autospec(KnowledgeGraph, instance=True)


@pytest.fixture
def mock_kg_handler(monkeypatch):
  mock = create_autospec(KnowledgeGraphHandler, instance=True)
  monkeypatch.setattr(
    "prometheus.neo4j.knowledge_graph_handler.KnowledgeGraphHandler", lambda *args, **kwargs: mock
  )
  return mock


def test_init_with_existing_graph(mock_neo4j_service, mock_kg_handler, mock_knowledge_graph):
  # Setup
  mock_kg_handler.knowledge_graph_exists.return_value = True
  mock_kg_handler.read_knowledge_graph.return_value = mock_knowledge_graph

  # Exercise
  service = KnowledgeGraphService(mock_neo4j_service, neo4j_batch_size=100)

  # Verify
  mock_kg_handler.knowledge_graph_exists.assert_called_once()
  mock_kg_handler.read_knowledge_graph.assert_called_once()
  assert service.kg == mock_knowledge_graph


def test_init_without_existing_graph(mock_neo4j_service, mock_kg_handler):
  # Setup
  mock_kg_handler.knowledge_graph_exists.return_value = False

  # Exercise
  service = KnowledgeGraphService(mock_neo4j_service, neo4j_batch_size=100)

  # Verify
  mock_kg_handler.knowledge_graph_exists.assert_called_once()
  mock_kg_handler.read_knowledge_graph.assert_not_called()
  assert service.kg is None


def test_clear_calls_handler_and_resets_kg(mock_neo4j_service, mock_kg_handler):
  # Setup
  mock_kg_handler.knowledge_graph_exists.return_value = True
  mock_kg_handler.read_knowledge_graph.return_value = mock_knowledge_graph

  # Exercise
  service = KnowledgeGraphService(mock_neo4j_service, neo4j_batch_size=100)
  service.clear()

  # Verify
  mock_kg_handler.clear_knowledge_graph.assert_called_once()
  assert service.kg is None
