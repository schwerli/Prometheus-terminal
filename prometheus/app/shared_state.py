import logging
import uuid
from pathlib import Path
from typing import Mapping, Optional, Sequence

from langchain_community.chat_models import ChatLiteLLM
from langgraph.checkpoint.postgres import PostgresSaver
from neo4j import GraphDatabase
from psycopg import Connection
from psycopg.rows import dict_row

from prometheus.configuration import config
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs import context_provider_subgraph, issue_answer_subgraph
from prometheus.neo4j import knowledge_graph_handler
from prometheus.utils import postgres_util


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
    self.model = ChatLiteLLM(**config.config["litellm"])
    self.postgres_conn = Connection.connect(
      config.config["postgres"]["db_uri"],
      autocommit=True,
      prepare_threshold=0,
      row_factory=dict_row,
    )
    self.checkpointer = PostgresSaver(self.postgres_conn)
    self.checkpointer.setup()
    self.cp_subgraph = None
    self.ia_subgraph = None
    self.git_repo = GitRepository(config.config["github"]["access_token"])

    self._logger = logging.getLogger("prometheus.app.shared_state")

    self._load_existing_knowledge_graph()

  def _load_existing_knowledge_graph(self):
    if self.kg_handler.knowledge_graph_exists():
      self.kg = self.kg_handler.read_knowledge_graph()
      self.cp_subgraph = context_provider_subgraph.ContextProviderSubgraph(
        self.model, self.kg, self.neo4j_driver, self.checkpointer
      )
      self.ia_subgraph = issue_answer_subgraph.IssueAnswerSubgraph(
        self.model, self.kg, self.neo4j_driver, self.checkpointer
      )

  def chat_with_context_provider(self, query: str, thread_id: Optional[str] = None):
    thread_id = str(uuid.uuid4()) if thread_id is None else thread_id
    response = self.cp_subgraph.invoke(query, thread_id)
    return thread_id, response

  def answer_issue(
    self,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    thread_id: Optional[str] = None,
  ):
    thread_id = str(uuid.uuid4()) if thread_id is None else thread_id
    response = self.ia_subgraph.invoke(issue_title, issue_body, issue_comments, thread_id)
    return response.content

  def get_all_conversation_ids(self) -> Sequence[str]:
    return postgres_util.get_all_thread_ids(self.checkpointer)

  def get_all_conversation_messages(self, conversation_id: str) -> Sequence[Mapping[str, str]]:
    return postgres_util.get_messages(self.checkpointer, conversation_id)

  def upload_local_repository(self, path: Path):
    kg = KnowledgeGraph(config.config["knowledge_graph"]["max_ast_depth"])
    kg.build_graph(path)
    self.kg = kg
    self.kg_handler.write_knowledge_graph(kg)
    self.cp_subgraph = context_provider_subgraph.ContextProviderSubgraph(
      self.model, self.kg, self.neo4j_driver, self.checkpointer
    )
    self.ia_subgraph = issue_answer_subgraph.IssueAnswerSubgraph(
      self.model, self.kg, self.neo4j_driver, self.checkpointer
    )

  def upload_github_repository(self, https_url: str):
    target_directory = Path(config.config["prometheus"]["working_directory"]) / "repositories"
    target_directory.mkdir(parents=True, exist_ok=True)
    saved_path = self.git_repo.clone_repository(https_url, target_directory)
    kg = KnowledgeGraph(config.config["knowledge_graph"]["max_ast_depth"])
    kg.build_graph(saved_path)
    self.kg = kg
    self.kg_handler.write_knowledge_graph(kg)
    self.cp_subgraph = context_provider_subgraph.ContextProviderSubgraph(
      self.model, self.kg, self.neo4j_driver, self.checkpointer
    )
    self.ia_subgraph = issue_answer_subgraph.IssueAnswerSubgraph(
      self.model, self.kg, self.neo4j_driver, self.checkpointer
    )

  def clear_knowledge_graph(self):
    self.kg_handler.clear_knowledge_graph()
    self.kg = None
    self.cp_subgraph = None
    self.ia_subgraph = None

    if self.git_repo.has_repository():
      self.git_repo.remove_repository()

  def has_knowledge_graph(self):
    return self.kg is not None

  def close(self):
    self.neo4j_driver.close()
    self.postgres_conn.close()
