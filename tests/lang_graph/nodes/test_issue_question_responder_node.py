
import pytest
from langchain_core.messages import HumanMessage

from prometheus.lang_graph.nodes.issue_question_responder_node import IssueQuestionResponderNode
from prometheus.lang_graph.subgraphs.issue_question_state import IssueQuestionState
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def fake_llm():
  return FakeListChatWithToolsModel(
    responses=["Based on the context, here's the answer to your question..."]
  )


@pytest.fixture
def basic_state():
  return IssueQuestionState(
    {
      "issue_title": "How to configure X?",
      "issue_body": "I need help configuring X in my project",
      "issue_comments": [
        {"username": "user1", "comment": "Have you tried Y?"},
        {"username": "user2", "comment": "I had the same question"},
      ],
      "question_context": "The user is asking about configuration options for X. Key settings include A, B, and C.",
    }
  )


def test_format_human_message_basic(fake_llm, basic_state):
  """Test basic human message formatting."""
  node = IssueQuestionResponderNode(fake_llm)
  message = node.format_human_message(basic_state)

  assert isinstance(message, HumanMessage)
  assert "How to configure X?" in message.content
  assert "I need help configuring X" in message.content
  assert "Have you tried Y?" in message.content
  assert "I had the same question" in message.content
  assert "configuration options for X" in message.content


def test_call_method(fake_llm, basic_state):
  """Test the call method execution."""
  node = IssueQuestionResponderNode(fake_llm)
  result = node(basic_state)

  assert "issue_response" in result
  assert result["issue_response"] == "Based on the context, here's the answer to your question..."


def test_empty_comments(fake_llm):
  """Test handling of empty comments."""
  state = IssueQuestionState(
    {
      "issue_title": "Question Title",
      "issue_body": "Question Body",
      "issue_comments": [],
      "question_context": "Analysis of the question",
    }
  )

  node = IssueQuestionResponderNode(fake_llm)
  message = node.format_human_message(state)

  assert "Comments:" in message.content
  assert message.content.count("username") == 0


def test_format_human_message_structure(fake_llm, basic_state):
  """Test the structure of the formatted human message."""
  node = IssueQuestionResponderNode(fake_llm)
  message = node.format_human_message(basic_state)

  assert "ISSUE INFORMATION:" in message.content
  assert "Title:" in message.content
  assert "Body:" in message.content
  assert "Comments:" in message.content
  assert "Retrieved question context:" in message.content
