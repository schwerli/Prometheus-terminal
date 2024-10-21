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
    neo4j_conversation_node = conversation.to_neo4j_conversation_node()
    with self.driver.session() as session:
      query = "CREATE (c:ConversationNode {conversation_id: $conversation_id, title: $title})"
      session.run(query, **neo4j_conversation_node)

  def delete_conversation(self, conversation_id: str):
    with self.driver.session() as session:
      query = """
        MATCH (c:ConversationNode {conversation_id: $conversation_id})
        OPTIONAL MATCH (c) -[:HAS_MESSAGE]-> (m:MessageNode)
        DETACH DELETE c, m
      """
      session.run(query, conversation_id=conversation_id)

  def delete_all_conversations(self):
    with self.driver.session() as session:
      query = """
        MATCH (n)
        WHERE n:ConversationNode OR n:MessageNode
        DETACH DELETE n
      """
      session.run(query)

  def add_message(self, conversation_id: str, message: message_types.Message):
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
        tx.run(query, **neo4j_message_node)

        query = """
        MATCH (c:ConversationNode {conversation_id: $conversation_id}), 
              (m:MessageNode {message_id: $message_id})
        MERGE (c) -[r:HAS_MESSAGE]-> (m)
        """
        tx.run(query, conversation_id=conversation_id, message_id=message.message_id)

        tx.commit()

  def load_conversation(self, conversation_id: str) -> Sequence[message_types.Message]:
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

  def get_all_conversation_id(self) -> Sequence[str]:
    conversation_ids = []
    with self.driver.session() as session:
      query = """
      MATCH (c:ConversationNode)
      RETURN c.conversation_id AS conversation_id
      """
      result = session.run(query)
      for record in result:
        conversation_ids.append(record["conversation_id"])
    return conversation_ids
