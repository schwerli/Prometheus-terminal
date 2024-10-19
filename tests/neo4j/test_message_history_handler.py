from datetime import datetime
import pytest
from neo4j import GraphDatabase
from testcontainers.neo4j import Neo4jContainer

from prometheus.message import message_types
from prometheus.neo4j.message_history_handler import MessageHistoryHandler


NEO4J_IMAGE = "neo4j:5.20.0"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

@pytest.fixture(scope="function")
def setup_container_and_handler():
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4J_PLUGINS", '["apoc"]')
  with container as neo4j_container:
    uri = neo4j_container.get_connection_url()
    handler = MessageHistoryHandler(uri, NEO4J_USERNAME, NEO4J_PASSWORD)
    yield neo4j_container, handler


def test_add_conversation(setup_container_and_handler):
  neo4j_container, handler = setup_container_and_handler
  uri = neo4j_container.get_connection_url()

  conversation_id = "foo"
  title = "bar"
  conversation = message_types.Conversation(conversation_id, title)
  handler.add_conversation(conversation)

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_conversation_node = session.run("MATCH (n:ConversationNode) RETURN n.conversation_id AS conversation_id, n.title AS title").data()
      assert len(read_conversation_node) == 1
      assert read_conversation_node[0]["conversation_id"] == conversation_id
      assert read_conversation_node[0]["title"] == title


def test_add_message(setup_container_and_handler):
  neo4j_container, handler = setup_container_and_handler
  uri = neo4j_container.get_connection_url()

  conversation_id = "1"
  title = "Hello world"
  conversation = message_types.Conversation(conversation_id, title)
  handler.add_conversation(conversation)

  message_id = "2"
  index = 0
  role = message_types.Role.user
  text = "Hi, how are you?"
  created_at = datetime.now()
  message = message_types.Message(message_id, index, role, text, created_at)
  handler.add_message(conversation_id, message)

  with GraphDatabase.driver(uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
    with driver.session() as session:
      read_message_node = session.run("MATCH (n:MessageNode) RETURN n.message_id AS message_id, n.index AS index, n.role AS role, n.text AS text, n.created_at AS created_at").data()
      assert len(read_message_node) == 1
      assert read_message_node[0]["message_id"] == message_id
      assert read_message_node[0]["index"] == index
      assert read_message_node[0]["role"] == role.value
      assert read_message_node[0]["text"] == text
      assert read_message_node[0]["created_at"] == created_at

      read_has_message_edge = session.run("MATCH (c:ConversationNode) -[:HAS_MESSAGE]-> (m:MessageNode) RETURN c.conversation_id AS conversation_id, m.message_id AS message_id").data()
      assert len(read_has_message_edge) == 1
      assert read_has_message_edge[0]["conversation_id"] == conversation_id
      assert read_has_message_edge[0]["message_id"] == message_id

def test_load_conversation(setup_container_and_handler):
  _, handler = setup_container_and_handler

  conversation_id = "1"
  title = "Hello world"
  conversation = message_types.Conversation(conversation_id, title)
  handler.add_conversation(conversation)

  message_id = "2"
  index = 0
  role = message_types.Role.user
  text = "Hi, how are you?"
  created_at = datetime.now()
  message = message_types.Message(message_id, index, role, text, created_at)
  handler.add_message(conversation_id, message)

  messages = handler.load_conversation(conversation_id)
  assert len(messages) == 1
  assert messages[0] == message