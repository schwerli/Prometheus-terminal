from unittest.mock import Mock, patch

import pytest

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode


@pytest.fixture
def mock_kg():
  kg = Mock(spec=KnowledgeGraph)
  kg.get_local_path.return_value = "/foo/bar"
  return kg


def test_git_diff_node(mock_kg):
  # Setup
  expected_diff = "sample diff content"

  with patch("prometheus.lang_graph.nodes.git_diff_node.GitRepository") as mock_git_repo_class:
    # Configure the mock GitRepository instance
    mock_git_repo_instance = Mock()
    mock_git_repo_instance.get_diff.return_value = expected_diff
    mock_git_repo_class.return_value = mock_git_repo_instance

    node = GitDiffNode(mock_kg)

    # Execute
    result = node({})

    # Assert
    assert result == {"patch": expected_diff}
    mock_git_repo_class.assert_called_once_with("/foo/bar", None, copy_to_working_dir=False)
    mock_git_repo_instance.get_diff.assert_called_once()
