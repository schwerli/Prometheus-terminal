from unittest.mock import Mock, patch

from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode


def test_git_diff_node():
  # Setup
  expected_diff = "sample diff content"
  fake_path = "/path/to/project"
  state = {
    "project_path": fake_path,
  }

  with patch("prometheus.lang_graph.nodes.git_diff_node.GitRepository") as mock_git_repo_class:
    # Configure the mock GitRepository instance
    mock_git_repo_instance = Mock()
    mock_git_repo_instance.get_diff.return_value = expected_diff
    mock_git_repo_class.return_value = mock_git_repo_instance

    node = GitDiffNode()

    # Execute
    result = node(state)

    # Assert
    assert result == {"patch": expected_diff}
    mock_git_repo_class.assert_called_once_with(fake_path, None, copy_to_working_dir=False)
    mock_git_repo_instance.get_diff.assert_called_once()
