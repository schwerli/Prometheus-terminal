from pathlib import Path
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
        "prometheus.neo4j.knowledge_graph_handler.KnowledgeGraphHandler",
        lambda *args, **kwargs: mock,
    )
    return mock


def test_build_and_save_new_knowledge_graph(mock_neo4j_service, mock_kg_handler, monkeypatch):
    # Setup
    mock_kg_handler.knowledge_graph_exists.return_value = False
    mock_kg_handler.get_new_knowledge_graph_root_node_id.return_value = 0

    # Mock KnowledgeGraph constructor and instance
    mock_kg = Mock()
    mock_kg_class = Mock(return_value=mock_kg)
    monkeypatch.setattr(
        "prometheus.app.services.knowledge_graph_service.KnowledgeGraph", mock_kg_class
    )
    mock_kg_class.build_graph.return_value = None
    mock_path = Path("/foo/bar")
    mock_kg.get_local_path.return_value = mock_path

    # Exercise
    service = KnowledgeGraphService(
        mock_neo4j_service,
        neo4j_batch_size=100,
        max_ast_depth=5,
        chunk_size=1000,
        chunk_overlap=100,
    )
    service.build_and_save_knowledge_graph(mock_path)

    # Verify
    mock_kg_class.assert_called_once_with(
        service.max_ast_depth,
        service.chunk_size,
        service.chunk_overlap,
        mock_kg_handler.get_new_knowledge_graph_root_node_id(),
    )
    mock_kg.build_graph.assert_called_once_with(mock_path)
    mock_kg_handler.write_knowledge_graph.assert_called_once_with(mock_kg)


def test_clear_knowledge_graph(mock_neo4j_service, mock_kg_handler):
    service = KnowledgeGraphService(
        mock_neo4j_service,
        neo4j_batch_size=100,
        max_ast_depth=5,
        chunk_size=1000,
        chunk_overlap=100,
    )

    # Act
    root_id = 123
    service.clear_kg(root_id)

    # Assert
    mock_kg_handler.clear_knowledge_graph.assert_called_once_with(root_id)
