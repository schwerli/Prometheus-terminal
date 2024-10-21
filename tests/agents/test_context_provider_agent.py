from neo4j import GraphDatabase

from prometheus.agents import context_provider_agent
from tests.test_utils.fixtures import neo4j_container_with_kg_fixture  # noqa: F401
from tests.test_utils.util import FakeListChatModelWithTools


def test_basic_context_provider_agent(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, kg = neo4j_container_with_kg_fixture
  fake_response = "fake response"
  fake_llm = FakeListChatModelWithTools(responses=[fake_response])
  with GraphDatabase.driver(
    neo4j_container.get_connection_url(), auth=(neo4j_container.username, neo4j_container.password)
  ) as driver:
    cp_agent = context_provider_agent.ContextProviderAgent(fake_llm, kg, driver)

    assert cp_agent.get_response("hello") == fake_response
