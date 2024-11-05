from unittest.mock import Mock, patch

from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


def test_git_diff_node():
    # Setup
    expected_diff = "sample diff content"
    mock_state = Mock(spec=IssueAnswerAndFixState)
    # Mock the state as a dictionary-like object
    mock_state.get = Mock(return_value="/path/to/project")
    mock_state.__getitem__ = Mock(return_value="/path/to/project")

    with patch('prometheus.lang_graph.nodes.git_diff_node.GitRepository') as mock_git_repo_class:
        # Configure the mock GitRepository instance
        mock_git_repo_instance = Mock()
        mock_git_repo_instance.get_diff.return_value = expected_diff
        mock_git_repo_class.return_value = mock_git_repo_instance

        node = GitDiffNode()

        # Execute
        result = node(mock_state)

        # Assert
        assert result == {"patch": expected_diff}
        mock_git_repo_class.assert_called_once_with("/path/to/project", None, copy_to_working_dir=False)
        mock_git_repo_instance.get_diff.assert_called_once()