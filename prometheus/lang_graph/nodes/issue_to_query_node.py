"""GitHub issue to query transformer for codebase analysis.

This module converts GitHub issues into structured queries suitable for processing
by AI agents. It combines issue titles, descriptions, and comments into a cohesive
query format that maintains the context and intent of the original issue.
"""

import logging
from typing import Mapping, Sequence

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueToQueryNode:
  """Transforms GitHub issues into structured queries for processing.

  This class takes GitHub issue information and formats it into a standardized
  query structure that can be processed by other components in the system. It
  maintains the original context while providing a consistent format for
  downstream processing.
  """

  def __init__(self):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.issue_to_query_node")

  def format_issue_comments(self, issue_comments: Sequence[Mapping[str, str]]):
    """Formats issue comments into a readable string.

    Converts a sequence of issue comments into a formatted string, preserving
    the username attribution for each comment.

    Args:
      issue_comments: Sequence of mappings containing comment information.
          Each mapping must have 'username' and 'comment' keys.

    Returns:
      Formatted string containing all comments with usernames, separated by newlines.
    """
    formatted_issue_comments = []
    for issue_comment in issue_comments:
      formatted_issue_comments.append(f"{issue_comment['username']}: {issue_comment['comment']}")
    return "\n\n".join(formatted_issue_comments)

  def __call__(self, state: IssueAnswerAndFixState):
    """Transforms the issue state into a structured query.

    Creates a formatted query string from the issue information in the state,
    including title, description, and any comments.

    Args:
      state: Current state containing issue information including title, body, and comments.

    Returns:
      Dictionary that update the state containing:
      - query: String containing the formatted query ready for processing
          by downstream components.
    """
    formatted_issue_comments = self.format_issue_comments(state["issue_comments"])
    query = f"""\
A user has reported the following issue to the codebase:
Title:
{state["issue_title"]}

Issue description: 
{state["issue_body"]}

Issue comments:
{formatted_issue_comments}

Now, please help the user with the issue.
"""

    self._logger.debug(f"Formatting \n{state}\n to \n{query}")
    return {"query": query}
