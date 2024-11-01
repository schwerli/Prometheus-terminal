from prometheus.graph.graph_types import MetadataNode
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j.knowledge_graph_handler import KnowledgeGraphHandler
from tests.test_utils import test_project_paths
from tests.test_utils.fixtures import (  # noqa: F401
  empty_neo4j_container_fixture,
  neo4j_container_with_kg_fixture,
)


def test_num_metadata_node(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture
  handler = KnowledgeGraphHandler(neo4j_container.get_driver(), 100)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_metadata_node = session.execute_read(handler._read_metadata_node)
      assert isinstance(read_metadata_node, MetadataNode)


def test_num_ast_nodes(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture
  handler = KnowledgeGraphHandler(neo4j_container.get_driver(), 100)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_ast_nodes = session.execute_read(handler._read_ast_nodes)
      assert len(read_ast_nodes) == 84


def test_num_file_nodes(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture
  handler = KnowledgeGraphHandler(neo4j_container.get_driver(), 100)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_file_nodes = session.execute_read(handler._read_file_nodes)
      assert len(read_file_nodes) == 9


def test_num_text_nodes(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture
  handler = KnowledgeGraphHandler(neo4j_container.get_driver(), 100)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_text_nodes = session.execute_read(handler._read_text_nodes)
      assert len(read_text_nodes) == 4


def test_num_parent_of_edges(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture
  handler = KnowledgeGraphHandler(neo4j_container.get_driver(), 100)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_parent_of_edges = session.execute_read(handler._read_parent_of_edges)
      assert len(read_parent_of_edges) == 81


def test_num_has_file_edges(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture
  handler = KnowledgeGraphHandler(neo4j_container.get_driver(), 100)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_has_file_edges = session.execute_read(handler._read_has_file_edges)
      assert len(read_has_file_edges) == 8


def test_num_has_ast_edges(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture
  handler = KnowledgeGraphHandler(neo4j_container.get_driver(), 100)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_has_ast_edges = session.execute_read(handler._read_has_ast_edges)
      assert len(read_has_ast_edges) == 3


def test_num_has_text_edges(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture
  handler = KnowledgeGraphHandler(neo4j_container.get_driver(), 100)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_has_text_edges = session.execute_read(handler._read_has_text_edges)
      assert len(read_has_text_edges) == 4


def test_num_next_chunk_edges(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture
  handler = KnowledgeGraphHandler(neo4j_container.get_driver(), 100)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_next_chunk_edges = session.execute_read(handler._read_next_chunk_edges)
      assert len(read_next_chunk_edges) == 3


def test_knowledge_graph_exists(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture
  handler = KnowledgeGraphHandler(neo4j_container.get_driver(), 100)

  assert handler.knowledge_graph_exists()


def test_clear_knowledge_graph(empty_neo4j_container_fixture):  # noqa: F811
  kg = KnowledgeGraph(1000)
  kg.build_graph(test_project_paths.TEST_PROJECT_PATH)

  driver = empty_neo4j_container_fixture.get_driver()
  handler = KnowledgeGraphHandler(driver, 100)
  handler.write_knowledge_graph(kg)

  assert handler.knowledge_graph_exists()

  handler.clear_knowledge_graph()

  assert not handler.knowledge_graph_exists()
