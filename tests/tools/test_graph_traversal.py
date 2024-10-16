from neo4j import GraphDatabase
import pytest

from testcontainers.neo4j import Neo4jContainer

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j.handler import Handler
from prometheus.tools import graph_traversal
from tests.test_utils import test_project_paths

NEO4J_IMAGE = "neo4j:5.20.0"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"


@pytest.fixture(scope="session")
def setup_neo4j_container():
  kg = KnowledgeGraph(test_project_paths.TEST_PROJECT_PATH, 1000)
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4JLABS_PLUGINS", '["apoc"]')
  with container as neo4j_container:
    uri = neo4j_container.get_connection_url()
    handler = Handler(uri, NEO4J_USERNAME, NEO4J_PASSWORD, "neo4j", 100)
    handler.write_knowledge_graph(kg)
    handler.close()
    yield neo4j_container


def test_find_file_node_with_basename(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.find_file_node_with_basename(
      test_project_paths.PYTHON_FILE.name, driver
    )

  basename = test_project_paths.PYTHON_FILE.name
  relative_path = str(
    test_project_paths.PYTHON_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  assert result.count("FileNode") == 1
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result


def test_find_file_node_with_relative_path(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  relative_path = str(
    test_project_paths.MD_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.find_file_node_with_relative_path(relative_path, driver)

  basename = test_project_paths.MD_FILE.name
  assert result.count("FileNode") == 1
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result


def test_find_ast_node_with_text(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  text = "System.out.println"
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.find_ast_node_with_text(text, driver)

  assert "FileNode" in result
  assert "ASTNode" in result

  basename = test_project_paths.JAVA_FILE.name
  relative_path = str(
    test_project_paths.JAVA_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert "'text': 'System.out.println(\"Hello world!\")'" in result
  assert "'type': 'method_invocation'" in result
  assert "'start_line': 2" in result
  assert "'end_line': 2" in result


def test_find_ast_node_with_type(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  type = "argument_list"
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.find_ast_node_with_type(type, driver)

  assert "FileNode" in result
  assert "ASTNode" in result

  basename = test_project_paths.JAVA_FILE.name
  relative_path = str(
    test_project_paths.JAVA_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert "'text': '(\"Hello world!\")'" in result
  assert f"'type': '{type}'" in result
  assert "'start_line': 2" in result
  assert "'end_line': 2" in result


def test_find_ast_node_with_text_in_file(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  text = "printf"
  basename = test_project_paths.C_FILE.name
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.find_ast_node_with_text_in_file(text, basename, driver)

  relative_path = str(
    test_project_paths.C_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert f"'text': '{text}'" in result
  assert "'type': 'identifier'" in result
  assert "'start_line': 3" in result
  assert "'end_line': 3" in result


def test_find_ast_node_with_type_in_file(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  type = "string_literal"
  basename = test_project_paths.C_FILE.name
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.find_ast_node_with_type_in_file(type, basename, driver)

  relative_path = str(
    test_project_paths.C_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert "'text': '\"Hello world!\"'" in result
  assert f"'type': '{type}'" in result
  assert "'start_line': 3" in result
  assert "'end_line': 3" in result


def test_find_ast_node_with_type_and_text(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  type = "string_literal"
  text = "Hello world!"
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.find_ast_node_with_type_and_text(type, text, driver)

  basename = test_project_paths.C_FILE.name
  relative_path = str(
    test_project_paths.C_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert f"'text': '\"{text}\"'" in result
  assert f"'type': '{type}'" in result
  assert "'start_line': 3" in result
  assert "'end_line': 3" in result


def test_find_text_node_with_text(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  text = "Text under header A."
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.find_text_node_with_text(text, driver)

  assert "FileNode" in result
  assert "TextNode" in result

  basename = test_project_paths.MD_FILE.name
  relative_path = str(
    test_project_paths.MD_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert f"'text': '{text}'" in result
  assert "'metadata': \"{'Header 1': 'A'}\"" in result


def test_find_text_node_with_text_in_file(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  text = "Text under header B."
  basename = test_project_paths.MD_FILE.name
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.find_text_node_with_text_in_file(text, basename, driver)

  assert "FileNode" in result
  assert "TextNode" in result

  relative_path = str(
    test_project_paths.MD_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert f"'text': '{text}'" in result
  assert "'metadata': \"{'Header 1': 'A', 'Header 2': 'B'}\"" in result


def test_get_next_text_node_with_node_id(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  # node_id of TextNode 'Text under header B.'
  node_id = 36
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.get_next_text_node_with_node_id(node_id, driver)

  assert "'text': 'Text under header C.'" in result
  assert "'metadata': \"{'Header 1': 'A', 'Header 2': 'C'}\"" in result


def test_preview_source_code_file_content_with_basename(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  basename = test_project_paths.C_FILE.name
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.preview_file_content_with_basename(basename, driver)

  assert "FileNode" in result
  assert "preview" in result

  source_code = test_project_paths.C_FILE.open().read()
  basename = test_project_paths.C_FILE.name
  relative_path = str(
    test_project_paths.C_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert source_code in result


def test_preview_text_file_content_with_basename(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  basename = test_project_paths.MD_FILE.name
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.preview_file_content_with_basename(basename, driver)

  assert "FileNode" in result
  assert "preview" in result

  basename = test_project_paths.MD_FILE.name
  relative_path = str(
    test_project_paths.MD_FILE.relative_to(
      test_project_paths.TEST_PROJECT_PATH
    ).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert "Text under header A." in result


def test_get_parent_node(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  node_id = 30
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.get_parent_node(node_id, driver)

  assert "ParentNode" in result
  assert "ASTNode" in result

  assert "'start_line': 2" in result
  assert "'end_line': 2" in result
  assert "'type': 'parameter_list'" in result
  assert "'text': '()'" in result


def test_get_children_node(setup_neo4j_container):
  neo4j_container = setup_neo4j_container
  uri = neo4j_container.get_connection_url()

  node_id = 20
  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    result = graph_traversal.get_children_node(node_id, driver)

  assert "ChildNode" in result
  assert "ASTNode" in result

  assert result.count("Result") == 3

  assert "'start_line': 3" in result
  assert "'end_line': 3" in result
  assert "'type': 'string_literal'" in result
  assert "'text': '\"Hello world!\"'" in result
