from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from prometheus.docker.python_container import PythonContainer
from prometheus.lang_graph.nodes.run_test_node import RunTestNode
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


@pytest.fixture
def mock_python_container():
  container_instance = Mock(spec=PythonContainer)
  container_instance.run_test.return_value = "Default test output"
  with patch("prometheus.lang_graph.nodes.run_test_node.PythonContainer", autospec=True) as mock:
    mock.return_value = container_instance
    yield mock, container_instance


@pytest.fixture
def mock_state():
  state = Mock(spec=IssueAnswerAndFixState)
  # Mock the dictionary-like access for project_path
  state.__getitem__ = Mock(return_value="/fake/path")
  return state


def test_run_test_node_before_tests(mock_python_container, mock_state):
  # Setup
  mock_constructor, mock_instance = mock_python_container
  test_output = "All tests passed successfully"
  mock_instance.run_test.return_value = test_output

  node = RunTestNode(test_state_attr="before_test_output")

  # Execute
  result = node(mock_state)

  # Assert
  assert result == {"before_test_output": test_output}
  mock_constructor.assert_called_once_with(Path("/fake/path"))
  mock_instance.run_test.assert_called_once()
  mock_instance.cleanup.assert_called_once()


def test_run_test_node_after_tests(mock_python_container, mock_state):
  # Setup
  mock_constructor, mock_instance = mock_python_container
  test_output = "All tests passed successfully"
  mock_instance.run_test.return_value = test_output

  node = RunTestNode(test_state_attr="after_test_output")

  # Execute
  result = node(mock_state)

  # Assert
  assert result == {"after_test_output": test_output}
  mock_constructor.assert_called_once_with(Path("/fake/path"))
  mock_instance.run_test.assert_called_once()
  mock_instance.cleanup.assert_called_once()


def test_run_test_node_failed_tests(mock_python_container, mock_state):
  # Setup
  mock_constructor, mock_instance = mock_python_container
  test_output = "Test failures occurred:\ntest_example.py::test_func FAILED"
  mock_instance.run_test.return_value = test_output

  node = RunTestNode(test_state_attr="before_test_output")

  # Execute
  result = node(mock_state)

  # Assert
  assert result == {"before_test_output": test_output}
  mock_constructor.assert_called_once_with(Path("/fake/path"))
  mock_instance.run_test.assert_called_once()
  mock_instance.cleanup.assert_called_once()
