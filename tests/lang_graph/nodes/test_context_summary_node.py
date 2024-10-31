from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from prometheus.lang_graph.nodes.context_summary_node import ContextSummaryNode
from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


def test_context_summary_node():
  fake_response = "Test summary response"
  fake_llm = FakeListChatModel(responses=[fake_response])
  node = ContextSummaryNode(model=fake_llm)

  test_messages = [
    AIMessage(content="This code handles file processing"),
    ToolMessage(content="Found implementation in utils.py", tool_call_id="test_tool_call_1"),
  ]
  test_state = ContextProviderState(
    {"query": "How does the error handling work?", "context_messages": test_messages}
  )

  result = node(test_state)

  assert "summary" in result
  assert isinstance(result["summary"], AIMessage)
  assert result["summary"].content == fake_response


def test_format_messages():
  fake_response = "Test summary response"
  fake_llm = FakeListChatModel(responses=[fake_response])
  node = ContextSummaryNode(model=fake_llm)

  test_messages = [
    AIMessage(content="This code handles file processing"),
    ToolMessage(content="Found implementation in utils.py", tool_call_id="test_tool_call_1"),
  ]

  # Verify the messages were formatted correctly
  formatted_messages = node.format_messages(test_messages)
  assert len(formatted_messages) == 2
  assert formatted_messages[0] == "Assistant message: This code handles file processing"
  assert formatted_messages[1] == "Tool message: Found implementation in utils.py"
