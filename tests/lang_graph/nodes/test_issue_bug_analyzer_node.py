import pytest
from langchain_core.messages import HumanMessage

from prometheus.lang_graph.nodes.issue_bug_analyzer_node import IssueBugAnalyzerNode
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def fake_llm():
  return FakeListChatWithToolsModel(responses=["Bug analysis completed successfully"])


def test_call_method_basic(fake_llm):
  """Test basic call functionality."""
  node = IssueBugAnalyzerNode(fake_llm)
  state = {"issue_bug_analyzer_messages": [HumanMessage(content="Please analyze this bug: ...")]}

  result = node(state)

  assert "issue_bug_analyzer_messages" in result
  assert len(result["issue_bug_analyzer_messages"]) == 1
  assert result["issue_bug_analyzer_messages"][0].content == "Bug analysis completed successfully"
