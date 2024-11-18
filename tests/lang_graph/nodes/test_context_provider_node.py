import pytest
from langchain_core.messages import AIMessage, ToolMessage

from prometheus.lang_graph.nodes.context_provider_node import ContextProviderNode
from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState
from tests.test_utils.fixtures import neo4j_container_with_kg_fixture  # noqa: F401
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.mark.slow
def test_context_provider_node(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, kg = neo4j_container_with_kg_fixture
  fake_response = "Fake response"
  fake_llm = FakeListChatWithToolsModel(responses=[fake_response])
  node = ContextProviderNode(fake_llm, kg, neo4j_container.get_driver(), 1000)

  test_messages = [
    AIMessage(content="This code handles file processing"),
    ToolMessage(content="Found implementation in utils.py", tool_call_id="test_tool_call_1"),
  ]
  test_state = ContextProviderState(
    {"query": "How does the error handling work?", "context_messages": test_messages}
  )

  result = node(test_state)

  assert "context_messages" in result
  assert result["context_messages"][0].content == fake_response
