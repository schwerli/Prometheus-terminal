from pathlib import Path
from typing import Mapping, Optional, Sequence

from prometheus.app.services.chat_service import ChatService
from prometheus.app.services.issue_answer_and_fix_service import IssueAnswerAndFixService
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
    github_token: str,
  ):
    self.knowledge_graph_service = knowledge_graph_service
    self.llm_service = llm_service
    self.neo4j_service = neo4j_service
    self.postgres_service = postgres_service
    self.repository_service = repository_service
    self.github_token = github_token

    self._initialize_subgraph_services()

  def _initialize_subgraph_services(self):
    self.chat_service = ChatService(
      self.knowledge_graph_service, self.neo4j_service, self.postgres_service, self.llm_service
    )
    if self.knowledge_graph_service.local_path is not None:
      self.issue_answer_and_fix_service = IssueAnswerAndFixService(
        self.knowledge_graph_service,
        self.neo4j_service,
        self.postgres_service,
        self.llm_service,
        Path(self.knowledge_graph_service.local_path),
      )
    else:
      self.issue_answer_and_fix_service = None

  def chat_with_codebase(self, query: str, thread_id: Optional[str] = None) -> tuple[str, str]:
    return self.chat_service.chat(query, thread_id)

  def answer_and_fix_issue(
    self,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    only_answer: bool,
    run_build: bool,
    run_tests: bool,
    thread_id: Optional[str] = None,
  ) -> str:
    issue_response, patch = self.issue_answer_and_fix_service.answer_and_fix_issue(
      issue_title, issue_body, issue_comments, only_answer, run_build, run_tests, thread_id
    )
    remote_branch_name = None
    if patch:
      remote_branch_name = self.repository_service.push_change_to_remote(f"Fixes #{issue_number}")
    return issue_response, remote_branch_name

  def exists_knowledge_graph(self) -> bool:
    return self.knowledge_graph_service.exists()

  def upload_local_repository(self, path: Path):
    self.knowledge_graph_service.clear()
    self.knowledge_graph_service.build_and_save_knowledge_graph(path)
    self._initialize_subgraph_services()

  def upload_github_repository(self, https_url: str, commit_id: Optional[str] = None):
    self.knowledge_graph_service.clear()
    saved_path = self.repository_service.clone_github_repo(self.github_token, https_url, commit_id)
    self.knowledge_graph_service.build_and_save_knowledge_graph(saved_path, https_url, commit_id)
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
