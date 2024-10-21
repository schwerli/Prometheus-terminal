import pytest
from langchain_core.messages import AIMessage, HumanMessage
from testcontainers.neo4j import Neo4jContainer

from prometheus.message import message_types
from prometheus.message.message_history import MessageHistory
from prometheus.neo4j.message_history_handler import MessageHistoryHandler

NEO4J_IMAGE = "neo4j:5.20.0"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"


@pytest.fixture(scope="function")
def setup_handler():
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4J_PLUGINS", '["apoc"]')
  with container as neo4j_container:
    uri = neo4j_container.get_connection_url()
    handler = MessageHistoryHandler(uri, NEO4J_USERNAME, NEO4J_PASSWORD)
    yield handler


def test_add_message(setup_handler):
  handler = setup_handler
  message_history = MessageHistory(handler)

  assert not handler.get_all_conversation_id()

  text = "Hello"
  message_history.add_message(message_types.Role.user, text)

  assert len(message_history.in_memory_message_history) == 1
  assert message_history.in_memory_message_history[0].text == text
  assert message_history.in_memory_message_history[0].role == message_types.Role.user
  assert len(handler.get_all_conversation_id()) == 1


def test_load_conversation(setup_handler):
  handler = setup_handler
  message_history_old = MessageHistory(handler)

  text1 = "random text 1"
  text2 = "random text 2"
  text3 = "random text 3"
  conversation_id = message_history_old.add_message(message_types.Role.user, text1)
  message_history_old.add_message(message_types.Role.user, text2)
  message_history_old.add_message(message_types.Role.user, text3)

  message_history_new = MessageHistory(handler)
  assert not message_history_new.in_memory_message_history

  message_history_new.load_conversation(conversation_id)
  assert len(message_history_new.in_memory_message_history) == 3
  assert message_history_new.in_memory_message_history[0].text == text1
  assert message_history_new.in_memory_message_history[1].text == text2
  assert message_history_new.in_memory_message_history[2].text == text3


def test_to_langchain_chat_history(setup_handler):
  handler = setup_handler
  message_history = MessageHistory(handler)

  text1 = "random text 1"
  text2 = "random text 2"
  text3 = "random text 3"
  message_history.add_message(message_types.Role.user, text1)
  message_history.add_message(message_types.Role.assistant, text2)
  message_history.add_message(message_types.Role.user, text3)

  langchain_chat_history = message_history.to_langchain_chat_history()
  assert len(langchain_chat_history) == 3
  assert isinstance(langchain_chat_history[0], HumanMessage)
  assert langchain_chat_history[0].content == text1
  assert isinstance(langchain_chat_history[1], AIMessage)
  assert langchain_chat_history[1].content == text2
  assert isinstance(langchain_chat_history[2], HumanMessage)
  assert langchain_chat_history[2].content == text3
