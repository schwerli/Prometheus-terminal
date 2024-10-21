from datetime import datetime

from prometheus.message import message_types
from prometheus.neo4j.message_history_handler import MessageHistoryHandler
from tests.test_utils.fixtures import empty_neo4j_container_fixture  # noqa: F401
from tests.test_utils.util import clean_neo4j_container


def test_add_conversation(empty_neo4j_container_fixture):  # noqa: F811
  neo4j_container = empty_neo4j_container_fixture
  handler = MessageHistoryHandler(neo4j_container.get_driver())

  conversation_id = "foo"
  title = "bar"
  conversation = message_types.Conversation(conversation_id, title)
  handler.add_conversation(conversation)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_conversation_node = session.run(
        "MATCH (n:ConversationNode) RETURN n.conversation_id AS conversation_id, n.title AS title"
      ).data()
      assert len(read_conversation_node) == 1
      assert read_conversation_node[0]["conversation_id"] == conversation_id
      assert read_conversation_node[0]["title"] == title

  clean_neo4j_container(neo4j_container)


def test_delete_conversation(empty_neo4j_container_fixture):  # noqa: F811
  neo4j_container = empty_neo4j_container_fixture
  handler = MessageHistoryHandler(neo4j_container.get_driver())

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

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_conversation_node = session.run(
        "MATCH (n:ConversationNode) RETURN n.conversation_id AS conversation_id, n.title AS title"
      ).data()
      assert len(read_conversation_node) == 1
      read_message_node = session.run(
        "MATCH (n:MessageNode) RETURN n.message_id AS message_id, n.index AS index, n.role AS role, n.text AS text, n.created_at AS created_at"
      ).data()
      assert len(read_message_node) == 1

      handler.delete_conversation(conversation_id)

      read_conversation_node = session.run(
        "MATCH (n:ConversationNode) RETURN n.conversation_id AS conversation_id, n.title AS title"
      ).data()
      assert len(read_conversation_node) == 0
      read_message_node = session.run(
        "MATCH (n:MessageNode) RETURN n.message_id AS message_id, n.index AS index, n.role AS role, n.text AS text, n.created_at AS created_at"
      ).data()
      assert len(read_message_node) == 0
  clean_neo4j_container(neo4j_container)


def test_delete_all_conversations(empty_neo4j_container_fixture):  # noqa: F811
  neo4j_container = empty_neo4j_container_fixture
  handler = MessageHistoryHandler(neo4j_container.get_driver())

  conversation_id_10 = "10"
  title_10 = "Hello world"
  conversation_10 = message_types.Conversation(conversation_id_10, title_10)
  handler.add_conversation(conversation_10)

  conversation_id_20 = "20"
  title_20 = "Hej världen"
  conversation_20 = message_types.Conversation(conversation_id_20, title_20)
  handler.add_conversation(conversation_20)

  message_id_1 = "1"
  index_1 = 0
  role_1 = message_types.Role.user
  text_1 = "Hi, how are you?"
  created_at_1 = datetime.now()
  message_1 = message_types.Message(message_id_1, index_1, role_1, text_1, created_at_1)
  handler.add_message(conversation_id_10, message_1)

  message_id_2 = "2"
  index_2 = 0
  role_2 = message_types.Role.assistant
  text_2 = "Hej, hur mår du?"
  created_at_2 = datetime.now()
  message_2 = message_types.Message(message_id_2, index_2, role_2, text_2, created_at_2)
  handler.add_message(conversation_id_20, message_2)

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_conversation_node = session.run(
        "MATCH (n:ConversationNode) RETURN n.conversation_id AS conversation_id, n.title AS title"
      ).data()
      assert len(read_conversation_node) == 2
      read_message_node = session.run(
        "MATCH (n:MessageNode) RETURN n.message_id AS message_id, n.index AS index, n.role AS role, n.text AS text, n.created_at AS created_at"
      ).data()
      assert len(read_message_node) == 2

      handler.delete_all_conversations()

      read_conversation_node = session.run(
        "MATCH (n:ConversationNode) RETURN n.conversation_id AS conversation_id, n.title AS title"
      ).data()
      assert len(read_conversation_node) == 0
      read_message_node = session.run(
        "MATCH (n:MessageNode) RETURN n.message_id AS message_id, n.index AS index, n.role AS role, n.text AS text, n.created_at AS created_at"
      ).data()
      assert len(read_message_node) == 0
  clean_neo4j_container(neo4j_container)


def test_add_message(empty_neo4j_container_fixture):  # noqa: F811
  neo4j_container = empty_neo4j_container_fixture
  handler = MessageHistoryHandler(neo4j_container.get_driver())

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

  with neo4j_container.get_driver() as driver:
    with driver.session() as session:
      read_message_node = session.run(
        "MATCH (n:MessageNode) RETURN n.message_id AS message_id, n.index AS index, n.role AS role, n.text AS text, n.created_at AS created_at"
      ).data()
      assert len(read_message_node) == 1
      assert read_message_node[0]["message_id"] == message_id
      assert read_message_node[0]["index"] == index
      assert read_message_node[0]["role"] == role.value
      assert read_message_node[0]["text"] == text
      assert read_message_node[0]["created_at"] == created_at

      read_has_message_edge = session.run(
        "MATCH (c:ConversationNode) -[:HAS_MESSAGE]-> (m:MessageNode) RETURN c.conversation_id AS conversation_id, m.message_id AS message_id"
      ).data()
      assert len(read_has_message_edge) == 1
      assert read_has_message_edge[0]["conversation_id"] == conversation_id
      assert read_has_message_edge[0]["message_id"] == message_id
  clean_neo4j_container(neo4j_container)


def test_load_conversation(empty_neo4j_container_fixture):  # noqa: F811
  neo4j_container = empty_neo4j_container_fixture
  handler = MessageHistoryHandler(neo4j_container.get_driver())

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
  clean_neo4j_container(neo4j_container)


def test_get_all_conversation_id(empty_neo4j_container_fixture):  # noqa: F811
  neo4j_container = empty_neo4j_container_fixture
  handler = MessageHistoryHandler(neo4j_container.get_driver())

  conversation_id_10 = "10"
  title_10 = "Hello world"
  conversation_10 = message_types.Conversation(conversation_id_10, title_10)
  handler.add_conversation(conversation_10)

  conversation_id_20 = "20"
  title_20 = "Hej världen"
  conversation_20 = message_types.Conversation(conversation_id_20, title_20)
  handler.add_conversation(conversation_20)

  read_conversation_ids = handler.get_all_conversation_id()

  assert len(read_conversation_ids) == 2
  assert conversation_id_10 in read_conversation_ids
  assert conversation_id_20 in read_conversation_ids
  clean_neo4j_container(neo4j_container)
