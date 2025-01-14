import pytest

from prometheus.tools import graph_traversal
from tests.test_utils import test_project_paths
from tests.test_utils.fixtures import neo4j_container_with_kg_fixture  # noqa: F401


@pytest.mark.slow
def test_find_file_node_with_basename(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_file_node_with_basename(
      test_project_paths.PYTHON_FILE.name, driver, 1000
    )

  basename = test_project_paths.PYTHON_FILE.name
  relative_path = str(
    test_project_paths.PYTHON_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )

  result_data = result[1]
  assert len(result_data) == 1
  assert "FileNode" in result_data[0]
  assert result_data[0]["FileNode"].get("basename", "") == basename
  assert result_data[0]["FileNode"].get("relative_path", "") == relative_path


@pytest.mark.slow
def test_find_file_node_with_relative_path(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, _ = neo4j_container_with_kg_fixture

  relative_path = str(
    test_project_paths.MD_FILE.relative_to(test_project_paths.TEST_PROJECT_PATH).as_posix()
  )
  with neo4j_container.get_driver() as driver:
    result = graph_traversal.find_file_node_with_relative_path(relative_path, driver, 1000)

  basename = test_project_paths.MD_FILE.name

  result_data = result[1]
  assert len(result_data) == 1
  assert "FileNode" in result_data[0]
  assert result_data[0]["FileNode"].get("basename", "") == basename
  assert result_data[0]["FileNode"].get("relative_path", "") == relative_path
