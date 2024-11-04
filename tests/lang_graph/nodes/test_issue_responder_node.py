from langchain_core.language_models import FakeListChatModel

from prometheus.lang_graph.nodes.issue_responder_node import IssueResponderNode


def test_format_human_message():
  fake_model = FakeListChatModel(responses=[""])

  issue_responder_node = IssueResponderNode(model=fake_model)

  issue_title = "Bug in user authentication"
  issue_body = "Users are unable to log in after the recent update."
  summary = "Check authentication.py around line 45 where the login logic was recently updated."
  user1 = "user1"
  comment1 = "I've experienced this issue as well."
  user2 = "user2"
  comment2 = "A potential fix is to adjust the memory settings."
  state = {
    "issue_title": issue_title,
    "issue_body": issue_body,
    "issue_comments": [{"username": user1, "comment": comment1}, {"username": user2, "comment": comment2}],
    "summary": summary,
  }

  expected_human_message = f"""\
Issue title:
{issue_title}

Issue body:
{issue_body}

Retrieved relevant context summary:
{summary}

Issue comments:
{user1}: {comment1}

{user2}: {comment2}
"""

  human_message = issue_responder_node.format_human_message(state).content
  assert human_message == expected_human_message


def test_issue_responder_node():
  fake_response = "Test response"
  fake_model = FakeListChatModel(responses=[fake_response])

  issue_responder_node = IssueResponderNode(model=fake_model)

  issue_title = "Bug in user authentication"
  issue_body = "Users are unable to log in after the recent update."
  summary = "Check authentication.py around line 45 where the login logic was recently updated."
  user1 = "user1"
  comment1 = "I've experienced this issue as well."
  user2 = "user2"
  comment2 = "A potential fix is to adjust the memory settings."
  state = {
    "issue_title": issue_title,
    "issue_body": issue_body,
    "issue_comments": [{"username": user1, "comment": comment1}, {"username": user2, "comment": comment2}],
    "summary": summary,
  }

  result = issue_responder_node(state)
  assert "issue_response" in result
  assert result["issue_response"].content == fake_response
