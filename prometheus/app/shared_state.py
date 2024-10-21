from pathlib import Path
from typing import Optional

from langchain_anthropic import ChatAnthropic
from neo4j import GraphDatabase

from prometheus.agents import context_provider_agent
from prometheus.configuration import config
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.message import message_types
from prometheus.message.message_history import MessageHistory
from prometheus.neo4j import knowledge_graph_handler, message_history_handler


class SharedState:
  def __init__(self):
    self.kg = None
    self.neo4j_driver = GraphDatabase.driver(
      config.config["neo4j"]["uri"],
      auth=(config.config["neo4j"]["username"], config.config["neo4j"]["password"]),
    )
    self.kg_handler = knowledge_graph_handler.KnowledgeGraphHandler(
      self.neo4j_driver,
      config.config["neo4j"]["batch_size"],
    )
    self.llm = ChatAnthropic(
      model=config.config["anthropic"]["model"],
      temperature=config.config["anthropic"]["temperature"],
      max_tokens=config.config["anthropic"]["max_tokens"],
      api_key=config.config["anthropic"]["api_key"],
    )
    self.cp_agent = None
    mh_handler = message_history_handler.MessageHistoryHandler(self.neo4j_driver)
    self.message_history = MessageHistory(mh_handler)

    self._load_existing_knowledge_graph()

  def _load_existing_knowledge_graph(self):
    if self.kg_handler.knowledge_graph_exists():
      self.kg = self.kg_handler.load_knowledge_graph()
      self.cp_agent = context_provider_agent.ContextProviderAgent(
        self.llm, self.kg, self.neo4j_driver
      )

  def chat_with_context_provider_agent(self, query: str, conversation_id: Optional[str] = None):
    if conversation_id:
      self.message_history.load_conversation(conversation_id)

    conversation_id = self.message_history.add_message(message_types.Role.user, query)
    response = self.cp_agent.get_response(query, self.message_history)
    self.message_history.add_message(message_types.Role.assistant, response)
    return conversation_id, response

  def upload_repository(self, path: Path):
    kg = KnowledgeGraph(config.config["knowledge_graph"]["max_ast_depth"])
    kg.build_graph(path)
    self.kg = kg
    self.kg_handler.write_knowledge_graph(kg)
    self.cp_agent = context_provider_agent.ContextProviderAgent(
      self.llm, self.kg, self.neo4j_driver
    )

  def clear_knowledge_graph(self):
    self.kg_handler.clear_knowledge_graph()
    self.kg = None
    self.cp_agent = None

  def has_knowledge_graph(self):
    return self.kg is not None

  def close(self):
    self.kg_handler.close()
    self.neo4j_driver.close()
