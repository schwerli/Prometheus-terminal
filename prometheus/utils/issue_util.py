from typing import Mapping, Sequence


def format_issue_comments(issue_comments: Sequence[Mapping[str, str]]):
  """Formats a sequence of issue comments into a readable string.

  Combines multiple issue comments with their associated usernames into a
  formatted string suitable for inclusion in the response context.

  Args:
    issue_comments: Sequence of mappings containing 'username' and 'comment'
      keys for each issue comment.

  Returns:
    Formatted string containing all comments with usernames, separated by newlines.
  """
  formatted_issue_comments = []
  for issue_comment in issue_comments:
    formatted_issue_comments.append(f"{issue_comment['username']}: {issue_comment['comment']}")
  return "\n\n".join(formatted_issue_comments)
