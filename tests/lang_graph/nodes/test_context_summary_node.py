from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from prometheus.lang_graph.nodes.context_summary_node import ContextSummaryNode
from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


def test_context_summary_node_basic():
  """Test basic functionality of the ContextSummaryNode."""
  fake_response = "Test summary response"
  fake_llm = FakeListChatModel(responses=[fake_response])
  node = ContextSummaryNode(model=fake_llm)

  test_messages = [
    AIMessage(content="This code handles file processing"),
    ToolMessage(content="Found implementation in utils.py", tool_call_id="test_tool_call_1"),
  ]
  test_state = ContextProviderState(
    {
      "original_query": "How does the error handling work?",
      "all_context_provider_responses": test_messages,
    }
  )

  result = node(test_state)

  assert "summary" in result
  assert result["summary"] == fake_response


def test_format_human_message():
  """Test the formatting of human messages with query and context."""
  fake_llm = FakeListChatModel(responses=["Test response"])
  node = ContextSummaryNode(model=fake_llm)

  test_messages = [
    AIMessage(content="First context message"),
    ToolMessage(content="Second context message", tool_call_id="test_tool_1"),
  ]
  test_state = ContextProviderState(
    {"original_query": "Test query", "all_context_provider_responses": test_messages}
  )

  human_message = node.format_human_message(test_state)

  assert isinstance(human_message, HumanMessage)
  assert "Test query" in human_message.content
  assert "First context message" in human_message.content
  assert "Second context message" in human_message.content
