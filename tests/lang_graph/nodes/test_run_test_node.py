from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from prometheus.docker.python_container import PythonContainer
from prometheus.lang_graph.nodes.run_test_node import RunTestNode
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


@pytest.fixture
def mock_python_container():
  with patch("prometheus.lang_graph.nodes.run_test_node.PythonContainer", autospec=True) as mock:
    container_instance = Mock(spec=PythonContainer)
    mock.return_value = container_instance
    yield container_instance


@pytest.fixture
def mock_state():
  state = Mock(spec=IssueAnswerAndFixState)
  # Set up the required attributes from IssueAnswerAndFixState
  state.patch = ""
  state.before_compile_output = ""
  state.before_test_output = ""
  state.after_compile_output = ""
  state.after_test_output = ""
  state.code_edit_messages = []
  return state


def test_run_test_node_before_tests(mock_python_container, mock_state):
  # Setup
  project_path = Path("/fake/path")
  test_output = "All tests passed successfully"
  mock_python_container.run_tests.return_value = test_output

  node = RunTestNode(project_path=project_path, test_state_attr="before_test_output")

  # Execute
  result = node(mock_state)

  # Assert
  assert result == {"before_test_output": test_output}
  mock_python_container.run_tests.assert_called_once()
  mock_python_container.cleanup.assert_called_once()


def test_run_test_node_after_tests(mock_python_container, mock_state):
  # Setup
  project_path = Path("/fake/path")
  test_output = "All tests passed successfully"
  mock_python_container.run_tests.return_value = test_output

  node = RunTestNode(project_path=project_path, test_state_attr="after_test_output")

  # Execute
  result = node(mock_state)

  # Assert
  assert result == {"after_test_output": test_output}
  mock_python_container.run_tests.assert_called_once()
  mock_python_container.cleanup.assert_called_once()


def test_run_test_node_failed_tests(mock_python_container, mock_state):
  # Setup
  project_path = Path("/fake/path")
  test_output = "Test failures occurred:\ntest_example.py::test_func FAILED"
  mock_python_container.run_tests.return_value = test_output

  node = RunTestNode(project_path=project_path, test_state_attr="before_test_output")

  # Execute
  result = node(mock_state)

  # Assert
  assert result == {"before_test_output": test_output}
  mock_python_container.run_tests.assert_called_once()
  mock_python_container.cleanup.assert_called_once()
