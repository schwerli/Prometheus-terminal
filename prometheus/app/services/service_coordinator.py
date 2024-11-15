"""Central coordinator for managing interactions between various prometheus services.

This coordinator orchestrates the interactions between multiple specialized services
including knowledge graph management, LLM operations, database connections, and
repository management. It provides a unified interface for codebase analysis,
issue handling, and conversation management.
"""

from pathlib import Path
from typing import Mapping, Optional, Sequence

from prometheus.app.services.issue_answer_and_fix_service import IssueAnswerAndFixService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import ResponseModeEnum


class ServiceCoordinator:
  """Coordinates operations between various prometheus services.

  This class serves as the central orchestrator for all service interactions,
  managing the lifecycle of various operations including codebase analysis,
  issue handling, and conversation management. It ensures proper initialization,
  coordination, and cleanup of all dependent services.
  """

  def __init__(
    self,
    knowledge_graph_service: KnowledgeGraphService,
    llm_service: LLMService,
    neo4j_service: Neo4jService,
    postgres_service: PostgresService,
    repository_service: RepositoryService,
    github_token: str,
  ):
    """Initializes the service coordinator with required services.

    Args:
        knowledge_graph_service: Service for knowledge graph operations.
        llm_service: Service for language model operations.
        neo4j_service: Service for Neo4j database operations.
        postgres_service: Service for PostgreSQL operations.
        repository_service: Service for repository management.
        github_token: GitHub access token for repository operations.
    """
    self.knowledge_graph_service = knowledge_graph_service
    self.llm_service = llm_service
    self.neo4j_service = neo4j_service
    self.postgres_service = postgres_service
    self.repository_service = repository_service
    self.github_token = github_token

  def answer_and_fix_issue(
    self,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    response_mode: ResponseModeEnum,
    run_build: bool,
    run_tests: bool,
    dockerfile_content: Optional[str] = None,
    image_name: Optional[str] = None,
    workdir: Optional[str] = None,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
    thread_id: Optional[str] = None,
  ) -> str:
    """Analyzes and optionally fixes a code issue.

    Args:
      issue_number: Issue identifier number.
      issue_title: Title of the issue.
      issue_body: Main description of the issue.
      issue_comments: Sequence of comment dictionaries related to the issue.
      response_mode: The mode of response: auto (automatically determine whether to fix),
        only_answer (provide answer without changes), or answer_and_fix (provide answer and fix code).
      run_build: If True, runs build validation on generated fix.
      run_tests: If True, runs tests on generated fix.
      dockerfile_content: User defined Dockerfile content for the containerized enviroment.
      image_name: User defined image to be pulled.
      workdir: User defined workdir for the containerized enviroment.
      build_commands: User defined build commands for the containerized enviroment.
      test_commands: User defined test commands for the containerized enviroment.
      thread_id: Optional identifier for conversation id (Not used right now).

    Returns:
      Tuple of (issue response text, remote branch name if fix was pushed).
    """
    issue_answer_and_fix_service = IssueAnswerAndFixService(
      self.knowledge_graph_service,
      self.neo4j_service,
      self.postgres_service,
      self.llm_service,
      self.knowledge_graph_service.local_path,
      dockerfile_content,
      image_name,
      workdir,
      build_commands,
      test_commands,
    )
    issue_response, patch = issue_answer_and_fix_service.answer_and_fix_issue(
      issue_title, issue_body, issue_comments, response_mode, run_build, run_tests, thread_id
    )
    remote_branch_name = None
    if patch:
      remote_branch_name = self.repository_service.push_change_to_remote(f"Fixes #{issue_number}")
    return issue_response, remote_branch_name

  def exists_knowledge_graph(self) -> bool:
    return self.knowledge_graph_service.exists()

  def upload_local_repository(self, path: Path):
    """Uploads a local repository.

    Args:
      path: Path to the local repository directory.
    """
    self.knowledge_graph_service.clear()
    self.knowledge_graph_service.build_and_save_knowledge_graph(path)

  def upload_github_repository(self, https_url: str, commit_id: Optional[str] = None):
    """Uploads a GitHub repository.

    Args:
        https_url: HTTPS URL of the GitHub repository.
        commit_id: Optional specific commit to analyze.
    """
    self.knowledge_graph_service.clear()
    saved_path = self.repository_service.clone_github_repo(self.github_token, https_url, commit_id)
    self.knowledge_graph_service.build_and_save_knowledge_graph(saved_path, https_url, commit_id)

  def get_all_conversation_ids(self) -> Sequence[str]:
    """Retrieves all conversation thread IDs.

    Returns:
      Sequence of conversation identifier strings.
    """
    return self.postgres_service.get_all_thread_ids()

  def get_messages(self, conversation_id: str) -> list[dict[str, str]]:
    """Retrieves messages for a specific conversation.

    Args:
      conversation_id: Unique identifier for the conversation thread.

    Returns:
      List of message dictionaries containing role and text.
    """
    return self.postgres_service.get_messages(conversation_id)

  def clear(self):
    """Clears all service state and working directories.

    Resets the knowledge graph, cleans repository working directory,
    and reinitializes subgraph services.
    """
    self.knowledge_graph_service.clear()
    self.repository_service.clean_working_directory()

  def close(self):
    """Closes all database connections and releases resources.

    This method should be called when the coordinator is no longer needed
    to ensure proper cleanup of database connections and resources.
    """
    self.neo4j_service.close()
    self.postgres_service.close()
