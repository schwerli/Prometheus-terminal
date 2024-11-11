from unittest.mock import Mock, create_autospec

import pytest

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j.knowledge_graph_handler import KnowledgeGraphHandler


@pytest.fixture
def mock_neo4j_service():
  service = create_autospec(Neo4jService, instance=True)
  service.neo4j_driver = Mock(name="mock_neo4j_driver")
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
  service = KnowledgeGraphService(mock_neo4j_service, neo4j_batch_size=100, max_ast_depth=5)

  # Verify
  mock_kg_handler.knowledge_graph_exists.assert_called_once()
  mock_kg_handler.read_knowledge_graph.assert_called_once()
  assert service.kg == mock_knowledge_graph


def test_init_without_existing_graph(mock_neo4j_service, mock_kg_handler):
  # Setup
  mock_kg_handler.knowledge_graph_exists.return_value = False

  # Exercise
  service = KnowledgeGraphService(mock_neo4j_service, neo4j_batch_size=100, max_ast_depth=5)

  # Verify
  mock_kg_handler.knowledge_graph_exists.assert_called_once()
  mock_kg_handler.read_knowledge_graph.assert_not_called()
  assert service.kg is None


def test_build_and_save_new_knowledge_graph(mock_neo4j_service, mock_kg_handler, monkeypatch):
  # Setup
  mock_kg_handler.knowledge_graph_exists.return_value = False

  # Mock KnowledgeGraph constructor and instance
  mock_kg = Mock()
  mock_kg_class = Mock(return_value=mock_kg)
  monkeypatch.setattr(
    "prometheus.app.services.knowledge_graph_service.KnowledgeGraph", mock_kg_class
  )
  mock_kg_class.build_graph.return_value = None
  mock_path = "/foo/bar"
  mock_kg.get_local_path.return_value = mock_path

  # Exercise
  service = KnowledgeGraphService(mock_neo4j_service, neo4j_batch_size=100, max_ast_depth=5)
  service.build_and_save_knowledge_graph(mock_path)

  # Verify
  mock_kg_class.assert_called_once_with(service.max_ast_depth)
  mock_kg.build_graph.assert_called_once_with(mock_path, None, None)
  mock_kg_handler.write_knowledge_graph.assert_called_once_with(mock_kg)
  assert service.kg == mock_kg


def test_build_and_save_clear_existing_knowledge_graph(
  mock_neo4j_service, mock_kg_handler, monkeypatch
):
  # Setup
  mock_kg_handler.knowledge_graph_exists.return_value = True

  # Mock KnowledgeGraph constructor and instance
  mock_kg = Mock()
  mock_kg_class = Mock(return_value=mock_kg)
  monkeypatch.setattr(
    "prometheus.app.services.knowledge_graph_service.KnowledgeGraph", mock_kg_class
  )
  mock_kg_class.build_graph.return_value = None
  mock_path = "/foo/bar"
  mock_kg.get_local_path.return_value = mock_path

  # Exercise
  service = KnowledgeGraphService(mock_neo4j_service, neo4j_batch_size=100, max_ast_depth=5)
  service.build_and_save_knowledge_graph(mock_path)

  # Verify
  mock_kg_handler.clear_knowledge_graph.assert_called_once()
  mock_kg_class.assert_called_once_with(service.max_ast_depth)
  mock_kg.build_graph.assert_called_once_with(mock_path, None, None)
  mock_kg_handler.write_knowledge_graph.assert_called_once_with(mock_kg)
  assert service.kg == mock_kg


def test_clear_calls_handler_and_resets_kg(
  mock_neo4j_service, mock_kg_handler, mock_knowledge_graph
):
  # Setup
  mock_kg_handler.knowledge_graph_exists.return_value = True
  mock_kg_handler.read_knowledge_graph.return_value = mock_knowledge_graph
  mock_knowledge_graph.get_local_path.return_value = "/mock/path"

  # Exercise
  service = KnowledgeGraphService(mock_neo4j_service, neo4j_batch_size=100, max_ast_depth=5)
  service.clear()

  # Verify
  mock_kg_handler.clear_knowledge_graph.assert_called_once()
  assert service.kg is None
  assert service.local_path is None
