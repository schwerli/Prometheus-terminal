import pytest
from neo4j import GraphDatabase
from testcontainers.neo4j import Neo4jContainer

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j.knowledge_graph_handler import KnowledgeGraphHandler
from tests.test_utils import test_project_paths

NEO4J_IMAGE = "neo4j:5.20.0"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"


@pytest.fixture(scope="module")
def setup_container_and_handler():
  kg = KnowledgeGraph(1000)
  kg.build_graph(test_project_paths.TEST_PROJECT_PATH)
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4J_PLUGINS", '["apoc"]')
  with container as neo4j_container:
    uri = neo4j_container.get_connection_url()
    handler = KnowledgeGraphHandler(uri, NEO4J_USERNAME, NEO4J_PASSWORD, 100)
    handler.write_knowledge_graph(kg)
    yield neo4j_container, handler


def test_num_ast_nodes(setup_container_and_handler):
  neo4j_container, handler = setup_container_and_handler
  uri = neo4j_container.get_connection_url()

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_ast_nodes = session.execute_read(handler._read_ast_nodes)
      assert len(read_ast_nodes) == 84


def test_num_file_nodes(setup_container_and_handler):
  neo4j_container, handler = setup_container_and_handler
  uri = neo4j_container.get_connection_url()

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_file_nodes = session.execute_read(handler._read_file_nodes)
      assert len(read_file_nodes) == 8


def test_num_text_nodes(setup_container_and_handler):
  neo4j_container, handler = setup_container_and_handler
  uri = neo4j_container.get_connection_url()

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_text_nodes = session.execute_read(handler._read_text_nodes)
      assert len(read_text_nodes) == 4


def test_num_parent_of_edges(setup_container_and_handler):
  neo4j_container, handler = setup_container_and_handler
  uri = neo4j_container.get_connection_url()

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_parent_of_edges = session.execute_read(handler._read_parent_of_edges)
      assert len(read_parent_of_edges) == 81


def test_num_has_file_edges(setup_container_and_handler):
  neo4j_container, handler = setup_container_and_handler
  uri = neo4j_container.get_connection_url()

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_has_file_edges = session.execute_read(handler._read_has_file_edges)
      assert len(read_has_file_edges) == 7


def test_num_has_ast_edges(setup_container_and_handler):
  neo4j_container, handler = setup_container_and_handler
  uri = neo4j_container.get_connection_url()

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_has_ast_edges = session.execute_read(handler._read_has_ast_edges)
      assert len(read_has_ast_edges) == 3


def test_num_has_text_edges(setup_container_and_handler):
  neo4j_container, handler = setup_container_and_handler
  uri = neo4j_container.get_connection_url()

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_has_text_edges = session.execute_read(handler._read_has_text_edges)
      assert len(read_has_text_edges) == 4


def test_num_next_chunk_edges(setup_container_and_handler):
  neo4j_container, handler = setup_container_and_handler
  uri = neo4j_container.get_connection_url()

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_next_chunk_edges = session.execute_read(handler._read_next_chunk_edges)
      assert len(read_next_chunk_edges) == 3


def test_knowledge_graph_exists(setup_container_and_handler):
  _, handler = setup_container_and_handler

  assert handler.knowledge_graph_exists()


def test_clear_knowledge_graph():
  kg = KnowledgeGraph(1000)
  kg.build_graph(test_project_paths.TEST_PROJECT_PATH)
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4J_PLUGINS", '["apoc"]')
  with container as neo4j_container:
    uri = neo4j_container.get_connection_url()
    handler = KnowledgeGraphHandler(uri, NEO4J_USERNAME, NEO4J_PASSWORD, 100)
    handler.write_knowledge_graph(kg)

    assert handler.knowledge_graph_exists()

    handler.clear_knowledge_graph()

    assert not handler.knowledge_graph_exists()
