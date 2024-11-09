from langchain_core.messages import AIMessage, ToolMessage

from prometheus.lang_graph.nodes.code_editing_node import CodeEditingNode
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from tests.test_utils.util import FakeListChatWithToolsModel


def test_code_editing_node():
  fake_response = "Fake response"
  fake_llm = FakeListChatWithToolsModel(responses=[fake_response])
  node = CodeEditingNode(fake_llm, "/foo/bar")

  test_messages = [
    AIMessage(content="Hi"),
    ToolMessage(content="Call tool", tool_call_id="test_tool_call_1"),
  ]
  test_state = IssueAnswerAndFixState(
    {
      "query": "Query",
      "summary": "Summary",
      "issue_title": "Issue title",
      "issue_body": "Issue body",
      "issue_comments": [],
      "project_path": "/foo/bar",
      "code_edit_messages": test_messages,
    }
  )

  result = node(test_state)

  assert "code_edit_messages" in result
  assert result["code_edit_messages"][0].content == fake_response
