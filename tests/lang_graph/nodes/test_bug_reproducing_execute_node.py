from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.bug_reproducing_execute_node import BugReproducingExecuteNode
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_container():
  return Mock(spec=BaseContainer)


@pytest.fixture
def mock_kg():
  kg = Mock(spec=KnowledgeGraph)
  kg.get_local_path.return_value = "/foo/bar"
  return kg


def test_bug_reproducing_execute_node_format_message(mock_container, mock_kg):
  fake_llm = FakeListChatWithToolsModel(responses=["test"])
  node = BugReproducingExecuteNode(fake_llm, mock_container, mock_kg)

  test_state = BugReproductionState(
    {
      "issue_title": "Test Bug",
      "issue_body": "Bug description",
      "issue_comments": ["Comment 1", "Comment 2"],
      "bug_context": "Context of the bug",
      "bug_reproducing_write_message": [AIMessage(content="Created test file at path/to/test.py")],
    }
  )

  message = node.format_human_message(test_state)

  assert isinstance(message, HumanMessage)
  assert "Test Bug" in message.content
  assert "Bug description" in message.content
  assert "Comment 1" in message.content
  assert "Created test file at path/to/test.py" in message.content


def test_bug_reproducing_execute_node_execution(mock_container, mock_kg):
  fake_response = "Test execution completed: All tests passed"
  fake_llm = FakeListChatWithToolsModel(responses=[fake_response])
  node = BugReproducingExecuteNode(fake_llm, mock_container, mock_kg)

  test_state = BugReproductionState(
    {
      "issue_title": "Test Bug",
      "issue_body": "Bug description",
      "issue_comments": ["Comment"],
      "bug_context": "Context of the bug",
      "bug_reproducing_write_message": [AIMessage(content="Created test file at path/to/test.py")],
      "bug_reproducing_execute_messages": [],
    }
  )

  result = node(test_state)

  assert "bug_reproducing_execute_messages" in result
  assert "last_bug_reproducing_execute_message" in result
  assert len(result["bug_reproducing_execute_messages"]) == 1
  assert result["bug_reproducing_execute_messages"][0].content == fake_response
  assert result["last_bug_reproducing_execute_message"].content == fake_response


def test_bug_reproducing_execute_node_with_existing_messages(mock_container, mock_kg):
  """Test execution with existing messages in state."""
  fake_response = "New test execution results"
  fake_llm = FakeListChatWithToolsModel(responses=[fake_response])
  node = BugReproducingExecuteNode(fake_llm, mock_container, mock_kg)

  existing_messages = [AIMessage(content="Previous execution result")]

  test_state = BugReproductionState(
    {
      "issue_title": "Test Bug",
      "issue_body": "Bug description",
      "issue_comments": [],
      "bug_context": "Context",
      "bug_reproducing_write_message": [AIMessage(content="Created test file at path/to/test.py")],
      "bug_reproducing_execute_messages": existing_messages,
    }
  )

  result = node(test_state)

  assert len(result["bug_reproducing_execute_messages"]) == 1
  assert result["bug_reproducing_execute_messages"][0].content == fake_response
  assert result["last_bug_reproducing_execute_message"].content == fake_response
