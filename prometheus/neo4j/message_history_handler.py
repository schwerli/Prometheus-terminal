from typing import Sequence

from neo4j import GraphDatabase

from prometheus.message import message_types


class MessageHistoryHandler:
  def __init__(self, driver: GraphDatabase.driver):
    """
    Args:
      driver: The neo4j driver.
    """
    self.driver = driver
    self._init_database()

    self._logger = logging.getLogger("prometheus.neo4j.message_history_handler")

  def _init_database(self):
    """Initialization of the neo4j database."""

    queries = [
      "CREATE CONSTRAINT unique_conversation_id IF NOT EXISTS "
      "FOR (n:ConversationNode) REQUIRE n.conversation_id IS UNIQUE",
      "CREATE CONSTRAINT unique_message_id IF NOT EXISTS "
      "FOR (n:MessageNode) REQUIRE n.message_id IS UNIQUE",
    ]
    with self.driver.session() as session:
      for query in queries:
        session.run(query)

  def add_conversation(self, conversation: message_types.Conversation):
    self._logger.info(f"Adding a new ConversationNode to neo4j: {conversation}")

    neo4j_conversation_node = conversation.to_neo4j_conversation_node()
    with self.driver.session() as session:
      query = "CREATE (c:ConversationNode {conversation_id: $conversation_id, title: $title})"
      session.run(query, **neo4j_conversation_node)

  def delete_conversation(self, conversation_id: str):
    self._logger.info(f"Deleting a ConversationNode from neo4j: {conversation_id}")
    with self.driver.session() as session:
      query = """
        MATCH (c:ConversationNode {conversation_id: $conversation_id})
        OPTIONAL MATCH (c) -[:HAS_MESSAGE]-> (m:MessageNode)
        DETACH DELETE c, m
      """
      session.run(query, conversation_id=conversation_id)

  def delete_all_conversations(self):
    self._logger.info("Deleting all ConversationNodes and MessageNodes from neo4j")
    with self.driver.session() as session:
      query = """
        MATCH (n)
        WHERE n:ConversationNode OR n:MessageNode
        DETACH DELETE n
      """
      session.run(query)

  def add_message(self, conversation_id: str, message: message_types.Message):
    self._logger.info(f"Adding a new MessageNode under ConversationNode {conversation_id} to neo4j: {message}")
    neo4j_message_node = message.to_neo4j_message_node()
    with self.driver.session() as session:
      with session.begin_transaction() as tx:
        query = """
        MERGE (m:MessageNode {message_id: $message_id})
        ON CREATE SET m.index = $index, 
                      m.role = $role, 
                      m.text = $text, 
                      m.created_at = $created_at
        """
        tx.run(
          query,
          message_id=neo4j_message_node["message_id"],
          index=neo4j_message_node["index"],
          role=neo4j_message_node["role"],
          text=neo4j_message_node["text"],
          created_at=neo4j_message_node["created_at"],
        )

        query = """
        MATCH (c:ConversationNode {conversation_id: $conversation_id}), 
              (m:MessageNode {message_id: $message_id})
        MERGE (c) -[r:HAS_MESSAGE]-> (m)
        """
        tx.run(query, conversation_id=conversation_id, message_id=message.message_id)

        tx.commit()

  def get_conversation_messages(self, conversation_id: str) -> Sequence[message_types.Message]:
    self._logger.info(f"Getting all MessageNodes under ConversationNode {conversation_id} from neo4j")
    messages = []
    with self.driver.session() as session:
      query = """
      MATCH (c:ConversationNode {conversation_id: $conversation_id}) -[:HAS_MESSAGE]-> (m:MessageNode)
      RETURN m.message_id AS message_id, m.index AS index, m.role AS role, 
             m.text AS text, m.created_at AS created_at
      """
      result = session.run(query, conversation_id=conversation_id)
      for record in result:
        messages.append(message_types.Message(**record))
    return messages

  def get_all_conversations(self) -> Sequence[message_types.Conversation]:
    self._logger.info("Getting all ConversationNodes from neo4j")
    conversations = []
    with self.driver.session() as session:
      query = """
      MATCH (c:ConversationNode)
      RETURN c.conversation_id AS conversation_id, c.title AS title
      """
      result = session.run(query)
      for record in result:
        conversations.append(message_types.Conversation(**record))
    return conversations
