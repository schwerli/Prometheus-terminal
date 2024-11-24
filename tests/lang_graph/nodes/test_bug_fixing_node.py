from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.bug_fixing_node import BugFixingNode
from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_kg():
  kg = Mock(spec=KnowledgeGraph)
  kg.get_local_path.return_value = "/test/path"
  return kg


@pytest.fixture
def fake_llm():
  return FakeListChatWithToolsModel(responses=["Bug fix analysis and implementation"])


@pytest.fixture
def basic_state():
  return IssueBugState(
    {
      "issue_title": "Test Bug",
      "issue_body": "There is a bug in the code",
      "issue_comments": [
        {"username": "user1", "comment": "Comment 1"},
        {"username": "user2", "comment": "Comment 2"},
      ],
      "bug_context": "Function xyz has an off-by-one error",
      "bug_fixing_messages": [],
      "reproduced_bug": True,
      "reproduced_bug_file": "test_file.py",
      "patch": None,
    }
  )


def test_format_human_message_basic(mock_kg, fake_llm, basic_state):
  """Test basic human message formatting."""
  node = BugFixingNode(fake_llm, mock_kg)
  message = node.format_human_message(basic_state)

  assert "Test Bug" in message
  assert "There is a bug in the code" in message
  assert "Comment 1" in message
  assert "Comment 2" in message
  assert "Function xyz has an off-by-one error" in message
  assert "test_file.py" in message


def test_format_human_message_with_patch(mock_kg, fake_llm, basic_state):
  """Test message formatting with existing patch."""
  state = basic_state.copy()
  state["patch"] = "Previous changes: modified line 42"

  node = BugFixingNode(fake_llm, mock_kg)
  message = node.format_human_message(state)

  assert "Previous changes: modified line 42" in message
  assert "You have previously made the following changes" in message


def test_format_human_message_without_reproduction(mock_kg, fake_llm, basic_state):
  """Test message formatting when bug is not reproduced."""
  state = basic_state.copy()
  state["reproduced_bug"] = False

  node = BugFixingNode(fake_llm, mock_kg)
  message = node.format_human_message(state)

  assert "BUG REPRODUCTION" not in message


def test_call_method(mock_kg, fake_llm, basic_state):
  """Test the __call__ method execution."""
  node = BugFixingNode(fake_llm, mock_kg)
  result = node(basic_state)

  assert "bug_fixing_messages" in result
  assert len(result["bug_fixing_messages"]) == 1
  assert result["bug_fixing_messages"][0].content == "Bug fix analysis and implementation"


def test_call_method_with_existing_messages(mock_kg, fake_llm, basic_state):
  """Test __call__ method with existing messages in state."""
  state = basic_state.copy()
  state["bug_fixing_messages"] = [AIMessage(content="Previous message")]

  node = BugFixingNode(fake_llm, mock_kg)
  result = node(state)

  assert len(result["bug_fixing_messages"]) == 1
  assert result["bug_fixing_messages"][0].content == "Bug fix analysis and implementation"
