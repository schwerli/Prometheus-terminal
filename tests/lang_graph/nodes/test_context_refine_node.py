from langchain_core.messages import AIMessage

from prometheus.lang_graph.nodes.context_refine_node import ContextRefineNode
from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState
from tests.test_utils.util import FakeListChatWithToolsModel


def test_context_refine_node_format_human_message_basic():
  """Test format_human_message without previous responses"""
  fake_model = FakeListChatWithToolsModel(responses=["Test response"])
  node = ContextRefineNode(fake_model)

  state = ContextProviderState(
    {
      "original_query": "How does auth work?",
      "context_provider_messages": [AIMessage(content="Found auth code")],
      "all_context_provider_responses": [],
    }
  )

  message = node.format_human_message(state)

  assert isinstance(message, str)
  assert "How does auth work?" in message
  assert "Found auth code" in message


def test_context_refine_node_format_human_message_with_history():
  """Test format_human_message with previous responses"""
  fake_model = FakeListChatWithToolsModel(responses=["Test response"])
  node = ContextRefineNode(fake_model)

  state = ContextProviderState(
    {
      "original_query": "How does auth work?",
      "context_provider_messages": [AIMessage(content="Current response")],
      "all_context_provider_responses": [
        AIMessage(content="Previous response 1"),
        AIMessage(content="Previous response 2"),
      ],
    }
  )

  message = node.format_human_message(state)

  assert "Previous response 1" in message
  assert "Previous response 2" in message
  assert "Current response" in message
