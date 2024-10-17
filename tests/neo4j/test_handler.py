import pytest
from neo4j import GraphDatabase
from testcontainers.neo4j import Neo4jContainer

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j.handler import Handler
from tests.test_utils import test_project_paths

NEO4J_IMAGE = "neo4j:5.20.0"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"


@pytest.fixture(scope="session")
def setup_neo4j_container():
  kg = KnowledgeGraph(test_project_paths.TEST_PROJECT_PATH, 1000)
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4J_PLUGINS", '["apoc"]')
  with container as neo4j_container:
    uri = neo4j_container.get_connection_url()
    handler = Handler(uri, NEO4J_USERNAME, NEO4J_PASSWORD, "neo4j", 100)
    handler.write_knowledge_graph(kg)
    handler.close()
    yield neo4j_container


def test_num_ast_nodes(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  def _count_num_ast_nodes(tx):
    result = tx.run("""
      MATCH (n:ASTNode)
      RETURN n
    """)
    values = [record.values() for record in result]
    return values

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_ast_nodes = session.execute_read(_count_num_ast_nodes)
      assert len(read_ast_nodes) == 84


def test_num_file_nodes(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  def _count_num_file_nodes(tx):
    result = tx.run("""
      MATCH (n:FileNode)
      RETURN n
    """)
    values = [record.values() for record in result]
    return values

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_file_nodes = session.execute_read(_count_num_file_nodes)
      assert len(read_file_nodes) == 8


def test_num_text_nodes(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  def _count_num_text_nodes(tx):
    result = tx.run("""
      MATCH (n:TextNode)
      RETURN n
    """)
    values = [record.values() for record in result]
    return values

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_text_nodes = session.execute_read(_count_num_text_nodes)
      assert len(read_text_nodes) == 4


def test_num_parent_of_edges(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  def _count_num_parent_of_edges(tx):
    result = tx.run("""
      MATCH () -[r:PARENT_OF]-> ()
      RETURN r
    """)
    values = [record.values() for record in result]
    return values

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_parent_of_edges = session.execute_read(_count_num_parent_of_edges)
      assert len(read_parent_of_edges) == 81


def test_num_has_file_edges(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  def _count_num_has_file_edges(tx):
    result = tx.run("""
      MATCH () -[r:HAS_FILE]-> ()
      RETURN r
    """)
    values = [record.values() for record in result]
    return values

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_has_file_edges = session.execute_read(_count_num_has_file_edges)
      assert len(read_has_file_edges) == 7


def test_num_has_ast_edges(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  def _count_num_has_ast_edges(tx):
    result = tx.run("""
      MATCH () -[r:HAS_AST]-> ()
      RETURN r
    """)
    values = [record.values() for record in result]
    return values

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_has_ast_edges = session.execute_read(_count_num_has_ast_edges)
      assert len(read_has_ast_edges) == 3


def test_num_has_text_edges(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  def _count_num_has_text_edges(tx):
    result = tx.run("""
      MATCH () -[r:HAS_TEXT]-> ()
      RETURN r
    """)
    values = [record.values() for record in result]
    return values

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_has_text_edges = session.execute_read(_count_num_has_text_edges)
      assert len(read_has_text_edges) == 4


def test_num_next_chunk_edges(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  def _count_num_next_chunk_edges(tx):
    result = tx.run("""
      MATCH () -[r:NEXT_CHUNK]-> ()
      RETURN r
    """)
    values = [record.values() for record in result]
    return values

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_next_chunk_edges = session.execute_read(_count_num_next_chunk_edges)
      assert len(read_next_chunk_edges) == 3
