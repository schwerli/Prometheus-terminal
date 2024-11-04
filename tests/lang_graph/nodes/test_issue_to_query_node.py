from prometheus.lang_graph.nodes.issue_to_query_node import IssueToQueryNode


def test_issue_to_query_node():
  issue_title = "Bug in data processing pipeline"
  issue_body = "The pipeline fails when processing large datasets."
  user1 = "user1"
  comment1 = "I've experienced this issue as well."
  user2 = "user2"
  comment2 = "A potential fix is to adjust the memory settings."
  state = {
    "issue_title": issue_title,
    "issue_body": issue_body,
    "issue_comments": [{"username": user1, "comment": comment1}, {"username": user2, "comment": comment2}],
  }

  expected_query = f"""\
A user has reported the following issue to the codebase:
Title:
{issue_title}

Issue description: 
{issue_body}

Issue comments:
{user1}: {comment1}

{user2}: {comment2}

Now, please help the user with the issue.
"""
  issue_to_query_node = IssueToQueryNode()
  result = issue_to_query_node(state)

  # Verify that the output query matches the expected query
  assert result["query"] == expected_query
