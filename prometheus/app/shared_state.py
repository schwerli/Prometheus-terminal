import logging
from pathlib import Path
from typing import Optional, Sequence

from langchain_community.chat_models import ChatLiteLLM
from neo4j import GraphDatabase

from prometheus.agents import context_provider_agent
from prometheus.configuration import config
from prometheus.git.git_repository import GitRepository
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
    self.llm = ChatLiteLLM(**config.config["litellm"])
    self.cp_agent = None
    mh_handler = message_history_handler.MessageHistoryHandler(self.neo4j_driver)
    self.message_history = MessageHistory(mh_handler)
    self.git_repo = GitRepository(config.config["github"]["access_token"])

    self._logger = logging.getLogger("prometheus.app.shared_state")

    self._load_existing_knowledge_graph()

  def _load_existing_knowledge_graph(self):
    if self.kg_handler.knowledge_graph_exists():
      self.kg = self.kg_handler.read_knowledge_graph()
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

  def get_all_conversations(self) -> Sequence[message_types.Conversation]:
    return self.message_history.get_all_conversations()

  def get_all_conversation_messages(self, conversation_id: str) -> Sequence[message_types.Message]:
    return [
      message.to_primitive_dict()
      for message in self.message_history.get_all_conversation_messages(conversation_id)
    ]

  def upload_local_repository(self, path: Path):
    kg = KnowledgeGraph(config.config["knowledge_graph"]["max_ast_depth"])
    kg.build_graph(path)
    self.kg = kg
    self.kg_handler.write_knowledge_graph(kg)
    self.cp_agent = context_provider_agent.ContextProviderAgent(
      self.llm, self.kg, self.neo4j_driver
    )

  def upload_github_repository(self, https_url: str):
    target_directory = Path(config.config["prometheus"]["working_directory"]) / "repositories"
    target_directory.mkdir(parents=True, exist_ok=True)
    saved_path = self.git_repo.clone_repository(https_url, target_directory)
    kg = KnowledgeGraph(config.config["knowledge_graph"]["max_ast_depth"])
    kg.build_graph(saved_path)
    self.kg = kg
    self.kg_handler.write_knowledge_graph(kg)
    self.cp_agent = context_provider_agent.ContextProviderAgent(
      self.llm, self.kg, self.neo4j_driver
    )

  def clear_knowledge_graph(self):
    self.kg_handler.clear_knowledge_graph()
    self.kg = None
    self.cp_agent = None

    if self.git_repo.has_repository():
      self.git_repo.remove_repository()

  def has_knowledge_graph(self):
    return self.kg is not None

  def close(self):
    self.neo4j_driver.close()
