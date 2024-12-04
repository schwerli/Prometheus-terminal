"""Central coordinator for managing interactions between various prometheus services.

This coordinator orchestrates the interactions between multiple specialized services
including knowledge graph management, LLM operations, database connections, and
repository management. It provides a unified interface for codebase analysis,
issue handling, and conversation management.
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

from prometheus.app.services.issue_service import IssueService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.lang_graph.graphs.issue_state import IssueType


class ServiceCoordinator:
  """Coordinates operations between various prometheus services.

  This class serves as the central orchestrator for all service interactions,
  managing the lifecycle of various operations including codebase analysis,
  issue handling, and conversation management. It ensures proper initialization,
  coordination, and cleanup of all dependent services.
  """

  def __init__(
    self,
    issue_service: IssueService,
    knowledge_graph_service: KnowledgeGraphService,
    llm_service: LLMService,
    neo4j_service: Neo4jService,
    postgres_service: PostgresService,
    repository_service: RepositoryService,
    max_token_per_neo4j_result: int,
    github_token: str,
    working_directory: Path,
  ):
    """Initializes the service coordinator with required services.

    Args:
      issue_service: Service for issue handling.
      knowledge_graph_service: Service for knowledge graph operations.
      llm_service: Service for language model operations.
      neo4j_service: Service for Neo4j database operations.
      postgres_service: Service for PostgreSQL operations.
      repository_service: Service for repository management.
      max_token_per_neo4j_result: Maximum number of tokens per Neo4j result.
      github_token: GitHub access token for repository operations.
      working_directory: Working directory for all Prometheus related files.
    """
    self.issue_service = issue_service
    self.knowledge_graph_service = knowledge_graph_service
    self.llm_service = llm_service
    self.neo4j_service = neo4j_service
    self.postgres_service = postgres_service
    self.repository_service = repository_service
    self.max_token_per_neo4j_result = max_token_per_neo4j_result
    self.github_token = github_token
    self.working_directory = working_directory
    self.answer_issue_log_dir = self.working_directory / "answer_issue_logs"
    self.answer_issue_log_dir.mkdir(parents=True, exist_ok=True)
    self._logger = logging.getLogger("prometheus.app.services.service_coordinator")

    if self.knowledge_graph_service.get_local_path() != self.repository_service.get_working_dir():
      self._logger.critical(
        f"Knowledge graph and repository working directories do not match: {self.knowledge_graph_service.get_local_path()} vs {self.repository_service.get_working_dir()}. Resetting all services."
      )
      self.clear()

  def answer_issue(
    self,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    issue_type: IssueType,
    run_build: bool,
    run_existing_test: bool,
    dockerfile_content: Optional[str] = None,
    image_name: Optional[str] = None,
    workdir: Optional[str] = None,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
    push_to_remote: Optional[bool] = None,
  ):
    logger = logging.getLogger("prometheus")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = self.answer_issue_log_dir / f"{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    try:
      issue_response, patch, reproduced_bug_file = self.issue_service.answer_issue(
        issue_title,
        issue_body,
        issue_comments,
        issue_type,
        run_build,
        run_existing_test,
        dockerfile_content,
        image_name,
        workdir,
        build_commands,
        test_commands,
      )

      remote_branch_name = None
      if patch and push_to_remote:
        remote_branch_name = self.repository_service.push_change_to_remote(
          f"Fixes #{issue_number}", [reproduced_bug_file]
        )
      return issue_response, patch, remote_branch_name
    except Exception as e:
      logger.error(f"Error in answer_issue: {str(e)}\n{traceback.format_exc()}")
    finally:
      logger.removeHandler(file_handler)
      file_handler.close()

  def exists_knowledge_graph(self) -> bool:
    return self.knowledge_graph_service.exists()

  def upload_local_repository(self, path: Path):
    """Uploads a local repository.

    Args:
      path: Path to the local repository directory.
    """
    self.clear()
    self.knowledge_graph_service.build_and_save_knowledge_graph(path)

  def upload_github_repository(self, https_url: str, commit_id: Optional[str] = None):
    """Uploads a GitHub repository.

    Args:
        https_url: HTTPS URL of the GitHub repository.
        commit_id: Optional specific commit to analyze.
    """
    self.clear()
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
    self.repository_service.clean()

  def close(self):
    """Closes all database connections and releases resources.

    This method should be called when the coordinator is no longer needed
    to ensure proper cleanup of database connections and resources.
    """
    self.neo4j_service.close()
    self.postgres_service.close()
