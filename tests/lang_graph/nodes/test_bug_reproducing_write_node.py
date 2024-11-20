from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.bug_reproducing_write_node import BugReproducingWriteNode
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_kg():
  kg = Mock(spec=KnowledgeGraph)
  kg.get_local_path.return_value = "/foo/bar"
  return kg


def test_bug_reproducing_write_node_format_message(mock_kg):
  """Test that human messages are formatted correctly with state data."""
  fake_llm = FakeListChatWithToolsModel(responses=["test"])
  node = BugReproducingWriteNode(fake_llm, mock_kg)

  test_state = BugReproductionState(
    {
      "issue_title": "Test Bug",
      "issue_body": "Bug description",
      "issue_comments": ["Comment 1", "Comment 2"],
      "bug_context": "Context of the bug",
      "reproduced_bug_file": "test/file.py",
      "last_bug_reproducing_execute_message": AIMessage(content="Previous attempt"),
      "bug_reproducing_write_messages": [],
    }
  )

  message = node.format_human_message(test_state)

  assert isinstance(message, HumanMessage)
  assert "Test Bug" in message.content
  assert "Bug description" in message.content
  assert "Comment 1" in message.content
  assert "test/file.py" in message.content
  assert "Previous attempt" in message.content


def test_bug_reproducing_write_node_execution(mock_kg):
  fake_response = "Created test file at path/to/test.py"
  fake_llm = FakeListChatWithToolsModel(responses=[fake_response])
  node = BugReproducingWriteNode(fake_llm, mock_kg)

  test_messages = [
    AIMessage(content="Previous message"),
    ToolMessage(content="Tool result", tool_call_id="test_call_1"),
  ]

  test_state = BugReproductionState(
    {
      "issue_title": "Test Bug",
      "issue_body": "Bug description",
      "issue_comments": ["Comment"],
      "bug_context": "Context of the bug",
      "reproduced_bug_file": "test/file.py",
      "last_bug_reproducing_execute_message": AIMessage(content="Previous attempt"),
      "bug_reproducing_write_messages": test_messages,
    }
  )

  result = node(test_state)

  assert "bug_reproducing_write_messages" in result
  assert len(result["bug_reproducing_write_messages"]) == 1
  assert result["bug_reproducing_write_messages"][0].content == fake_response
