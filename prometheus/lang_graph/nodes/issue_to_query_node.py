from typing import Mapping, Sequence

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueToQueryNode:
  def format_issue_comments(self, issue_comments: Sequence[Mapping[str, str]]):
    formatted_issue_comments = []
    for issue_comment in issue_comments:
      formatted_issue_comments.append(f"{issue_comment['username']}: {issue_comment['comment']}")
    return "\n\n".join(formatted_issue_comments)

  def __call__(self, state: IssueAnswerAndFixState):
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

    return {"query": query}
