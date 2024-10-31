from langchain_core.language_models import FakeListChatModel

from prometheus.lang_graph.nodes.issue_responder_node import IssueResponderNode


def test_format_human_message():
  fake_model = FakeListChatModel(responses=[""])

  issue_responder_node = IssueResponderNode(model=fake_model)

  issue_title = "Bug in user authentication"
  issue_body = "Users are unable to log in after the recent update."
  summary = "Check authentication.py around line 45 where the login logic was recently updated."
  state = {"issue_title": issue_title, "issue_body": issue_body, "summary": summary}

  expected_human_message = f"""\
Issue title:
{issue_title}

Issue body:
{issue_body}

Retrieved relevant context summary:
{summary}
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
  state = {"issue_title": issue_title, "issue_body": issue_body, "summary": summary}

  result = issue_responder_node(state)
  assert "issue_response" in result
  assert result["issue_response"].content == fake_response
