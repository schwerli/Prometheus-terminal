from pathlib import Path
from typing import Mapping, Optional, Sequence

from prometheus.app.services.chat_service import ChatService
from prometheus.app.services.issue_service import IssueService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.app.services.repository_service import RepositoryService


class ServiceCoordinator:
  def __init__(
    self,
    knowledge_graph_service: KnowledgeGraphService,
    llm_service: LLMService,
    neo4j_service: Neo4jService,
    postgres_service: PostgresService,
    repository_service: RepositoryService,
  ):
    self.knowledge_graph_service = knowledge_graph_service
    self.llm_service = llm_service
    self.neo4j_service = neo4j_service
    self.postgres_service = postgres_service
    self.repository_service = repository_service

    self._initialize_subgraph_services()

  def _initialize_subgraph_services(self):
    self.chat_service = ChatService(
      self.knowledge_graph_service, self.neo4j_service, self.postgres_service, self.llm_service
    )
    self.issue_service = IssueService(
      self.knowledge_graph_service, self.neo4j_service, self.postgres_service, self.llm_service
    )

  def chat_with_codebase(self, query: str, thread_id: Optional[str] = None) -> tuple[str, str]:
    return self.chat_service.chat(query, thread_id)

  def answer_issue(
    self,
    title: str,
    body: str,
    comments: Sequence[Mapping[str, str]],
    thread_id: Optional[str] = None,
  ) -> str:
    return self.issue_service.answer_issue(title, body, comments, thread_id)

  def exists_knowledge_graph(self) -> bool:
    return self.knowledge_graph_service.exists()

  def upload_local_repository(self, path: Path):
    self.knowledge_graph_service.clear()
    self.knowledge_graph_service.build_and_save_knowledge_graph(path)
    self._initialize_subgraph_services()

  def upload_github_repository(self, https_url: str, commit_id: Optional[str] = None):
    self.knowledge_graph_service.clear()
    saved_path = self.repository_service.clone_github_repo(https_url, commit_id)
    self.knowledge_graph_service.build_and_save_knowledge_graph(saved_path)
    self._initialize_subgraph_services()

  def get_all_conversation_ids(self) -> Sequence[str]:
    return self.postgres_service.get_all_thread_ids()

  def get_messages(self, conversation_id: str) -> list[dict[str, str]]:
    return self.postgres_service.get_messages(conversation_id)

  def clear(self):
    self.knowledge_graph_service.clear()
    self.repository_service.clean_working_directory()
    self._initialize_subgraph_services()

  def close(self):
    self.neo4j_service.close()
    self.postgres_service.close()
