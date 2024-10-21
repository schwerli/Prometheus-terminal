from prometheus.tools import graph_traversal
from tests.test_utils import test_project_paths
from tests.test_utils.fixtures import neo4j_container_with_kg_fixture  # noqa: F401


def test_find_file_node_with_basename(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_file_node_with_basename(
      test_project_paths.PYTHON_FILE.name, driver
    )

  basename = test_project_paths.PYTHON_FILE.name
  relative_path = str(
    test_project_paths.PYTHON_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  assert result.count("FileNode") == 1
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result


def test_find_file_node_with_relative_path(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  relative_path = str(
    test_project_paths.MD_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_file_node_with_relative_path(relative_path, driver)

  basename = test_project_paths.MD_FILE.name
  assert result.count("FileNode") == 1
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result


def test_find_ast_node_with_text(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  text = "System.out.println"
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_ast_node_with_text(text, driver)

  assert "FileNode" in result
  assert "ASTNode" in result

  basename = test_project_paths.JAVA_FILE.name
  relative_path = str(
    test_project_paths.JAVA_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert "'text': 'System.out.println(\"Hello world!\")'" in result
  assert "'type': 'method_invocation'" in result
  assert "'start_line': 2" in result
  assert "'end_line': 2" in result


def test_find_ast_node_with_type(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  type = "argument_list"
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_ast_node_with_type(type, driver)

  assert "FileNode" in result
  assert "ASTNode" in result

  basename = test_project_paths.JAVA_FILE.name
  relative_path = str(
    test_project_paths.JAVA_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert "'text': '(\"Hello world!\")'" in result
  assert f"'type': '{type}'" in result
  assert "'start_line': 2" in result
  assert "'end_line': 2" in result


def test_find_ast_node_with_text_in_file(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  text = "printf"
  basename = test_project_paths.C_FILE.name
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_ast_node_with_text_in_file(text, basename, driver)

  relative_path = str(
    test_project_paths.C_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert f"'text': '{text}'" in result
  assert "'type': 'identifier'" in result
  assert "'start_line': 3" in result
  assert "'end_line': 3" in result


def test_find_ast_node_with_type_in_file(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  type = "string_literal"
  basename = test_project_paths.C_FILE.name
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_ast_node_with_type_in_file(type, basename, driver)

  relative_path = str(
    test_project_paths.C_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert "'text': '\"Hello world!\"'" in result
  assert f"'type': '{type}'" in result
  assert "'start_line': 3" in result
  assert "'end_line': 3" in result


def test_find_ast_node_with_type_and_text(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  type = "string_literal"
  text = "Hello world!"
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_ast_node_with_type_and_text(type, text, driver)

  basename = test_project_paths.C_FILE.name
  relative_path = str(
    test_project_paths.C_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert f"'text': '\"{text}\"'" in result
  assert f"'type': '{type}'" in result
  assert "'start_line': 3" in result
  assert "'end_line': 3" in result


def test_find_text_node_with_text(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  text = "Text under header A."
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_text_node_with_text(text, driver)

  assert "FileNode" in result
  assert "TextNode" in result

  basename = test_project_paths.MD_FILE.name
  relative_path = str(
    test_project_paths.MD_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert f"'text': '{text}'" in result
  assert "'metadata': \"{'Header 1': 'A'}\"" in result


def test_find_text_node_with_text_in_file(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  text = "Text under header B."
  basename = test_project_paths.MD_FILE.name
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_text_node_with_text_in_file(text, basename, driver)

  assert "FileNode" in result
  assert "TextNode" in result

  relative_path = str(
    test_project_paths.MD_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert f"'text': '{text}'" in result
  assert "'metadata': \"{'Header 1': 'A', 'Header 2': 'B'}\"" in result


def test_get_next_text_node_with_node_id(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  # node_id of TextNode 'Text under header B.'
  node_id = 36
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.get_next_text_node_with_node_id(node_id, driver)

  assert "'text': 'Text under header C.'" in result
  assert "'metadata': \"{'Header 1': 'A', 'Header 2': 'C'}\"" in result


def test_preview_source_code_file_content_with_basename(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  basename = test_project_paths.C_FILE.name
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.preview_file_content_with_basename(basename, driver)

  assert "FileNode" in result
  assert "preview" in result

  source_code = test_project_paths.C_FILE.open().read()
  basename = test_project_paths.C_FILE.name
  relative_path = str(
    test_project_paths.C_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert source_code in result


def test_preview_text_file_content_with_basename(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  basename = test_project_paths.MD_FILE.name
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.preview_file_content_with_basename(basename, driver)

  assert "FileNode" in result
  assert "preview" in result

  basename = test_project_paths.MD_FILE.name
  relative_path = str(
    test_project_paths.MD_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  assert f"'basename': '{basename}'" in result
  assert f"'relative_path': '{relative_path}'" in result
  assert "Text under header A." in result


def test_get_parent_node(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  node_id = 30
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.get_parent_node(node_id, driver)

  assert "ParentNode" in result
  assert "ASTNode" in result

  assert "'start_line': 2" in result
  assert "'end_line': 2" in result
  assert "'type': 'parameter_list'" in result
  assert "'text': '()'" in result


def test_get_children_node(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  node_id = 20
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.get_children_node(node_id, driver)

  assert "ChildNode" in result
  assert "ASTNode" in result

  assert result.count("Result") == 3

  assert "'start_line': 3" in result
  assert "'end_line': 3" in result
  assert "'type': 'string_literal'" in result
  assert "'text': '\"Hello world!\"'" in result
