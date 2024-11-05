from langchain_core.messages import AIMessage, ToolMessage

from prometheus.lang_graph.nodes.code_editing_node import CodeEditingNode
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from tests.test_utils.util import FakeListChatWithToolsModel


def test_code_editing_node_without_test_output():
  fake_response = "Fake response"
  fake_llm = FakeListChatWithToolsModel(responses=[fake_response])
  node = CodeEditingNode(fake_llm, "/foo/bar")

  test_messages = [
    AIMessage(content="This code handles file processing"),
    ToolMessage(content="Found implementation in utils.py", tool_call_id="test_tool_call_1"),
  ]
  test_state = IssueAnswerAndFixState(
    {
      "query": "How does the error handling work?",
      "summary": "Test summary response",
      "code_edit_messages": test_messages,
    }
  )

  result = node(test_state)

  assert "code_edit_messages" in result
  assert result["code_edit_messages"][0].content == fake_response


def test_code_editing_node_with_test_output():
  fake_response = "Fake response"
  fake_llm = FakeListChatWithToolsModel(responses=[fake_response])
  node = CodeEditingNode(fake_llm, "/foo/bar")

  test_messages = [
    AIMessage(content="This code handles file processing"),
    ToolMessage(content="Found implementation in utils.py", tool_call_id="test_tool_call_1"),
  ]
  test_state = IssueAnswerAndFixState(
    {
      "query": "How does the error handling work?",
      "summary": "Test summary response",
      "code_edit_messages": test_messages,
      "test_output": "Test output",
    }
  )

  result = node(test_state)

  assert "code_edit_messages" in result
  assert result["code_edit_messages"][0].content == fake_response
