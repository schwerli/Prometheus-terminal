from unittest.mock import Mock

from prometheus.git.git_repository import GitRepository
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


def test_git_diff_node():
  # Setup
  mock_git_repo = Mock(spec=GitRepository)
  expected_diff = "sample diff content"
  mock_git_repo.get_diff.return_value = expected_diff

  node = GitDiffNode(git_repo=mock_git_repo)

  # Create a mock state
  mock_state = Mock(spec=IssueAnswerAndFixState)

  # Execute
  result = node(mock_state)

  # Assert
  assert result == {"patch": expected_diff}
  mock_git_repo.get_diff.assert_called_once()
