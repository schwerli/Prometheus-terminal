import pytest
from neo4j import GraphDatabase
from testcontainers.neo4j import Neo4jContainer

from prometheus.agents import chat_history, context_provider_agent, message_types
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j.handler import Handler
from tests.test_utils import test_project_paths
from tests.test_utils.llm import FakeListChatModelWithTools

NEO4J_IMAGE = "neo4j:5.20.0"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"


@pytest.fixture(scope="session")
def setup_neo4j_container_with_kg():
  kg = KnowledgeGraph(test_project_paths.TEST_PROJECT_PATH, 1000)
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4J_PLUGINS", '["apoc"]')
  with container as neo4j_container:
    uri = neo4j_container.get_connection_url()
    handler = Handler(uri, NEO4J_USERNAME, NEO4J_PASSWORD, "neo4j", 100)
    handler.write_knowledge_graph(kg)
    handler.close()
    yield kg, neo4j_container


def test_basic_context_provider_agent(setup_neo4j_container_with_kg):
  kg, neo4j_container = setup_neo4j_container_with_kg
  fake_response = "fake response"
  fake_llm = FakeListChatModelWithTools(responses=[fake_response])
  with GraphDatabase.driver(
    neo4j_container.get_connection_url(), auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
  ) as driver:
    cp_agent = context_provider_agent.ContextProviderAgent(fake_llm, kg, driver)

    messages = chat_history.ChatHistory(10)
    messages.add_message(
      message_types.Message(role=message_types.Role.user, message="hello")
    )
    assert cp_agent.get_response(messages) == fake_response
