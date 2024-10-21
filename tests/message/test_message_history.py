from langchain_core.messages import AIMessage, HumanMessage

from prometheus.message import message_types
from prometheus.message.message_history import MessageHistory
from prometheus.neo4j.message_history_handler import MessageHistoryHandler
from tests.test_utils.fixtures import empty_neo4j_container_fixture  # noqa: F401


def test_add_message(empty_neo4j_container_fixture):  # noqa: F811
  handler = MessageHistoryHandler(empty_neo4j_container_fixture.get_driver())
  message_history = MessageHistory(handler)

  assert not handler.get_all_conversation_id()

  text = "Hello"
  message_history.add_message(message_types.Role.user, text)

  assert len(message_history.in_memory_message_history) == 1
  assert message_history.in_memory_message_history[0].text == text
  assert message_history.in_memory_message_history[0].role == message_types.Role.user
  assert len(handler.get_all_conversation_id()) == 1


def test_load_conversation(empty_neo4j_container_fixture):  # noqa: F811
  handler = MessageHistoryHandler(empty_neo4j_container_fixture.get_driver())
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


def test_to_langchain_chat_history(empty_neo4j_container_fixture):  # noqa: F811
  handler = MessageHistoryHandler(empty_neo4j_container_fixture.get_driver())
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
