"""Central coordinator for managing interactions between various prometheus services.

This coordinator orchestrates the interactions between multiple specialized services
including knowledge graph management, LLM operations, database connections, and
repository management. It provides a unified interface for codebase analysis,
issue handling, and conversation management.
"""

from pathlib import Path
from typing import Optional, Sequence

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.app.services.repository_service import RepositoryService


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
    max_token_per_neo4j_result: int,
    postgres_service: PostgresService,
    repository_service: RepositoryService,
    github_token: str,
    working_directory: str,
  ):
    """Initializes the service coordinator with required services.

    Args:
        knowledge_graph_service: Service for knowledge graph operations.
        llm_service: Service for language model operations.
        neo4j_service: Service for Neo4j database operations.
        max_token_per_neo4j_result: Maximum number of tokens per Neo4j result.
        postgres_service: Service for PostgreSQL operations.
        repository_service: Service for repository management.
        github_token: GitHub access token for repository operations.
        working_directory: Working directory for all Prometheus related files.
    """
    self.knowledge_graph_service = knowledge_graph_service
    self.llm_service = llm_service
    self.neo4j_service = neo4j_service
    self.max_token_per_neo4j_result = max_token_per_neo4j_result
    self.postgres_service = postgres_service
    self.repository_service = repository_service
    self.github_token = github_token
    self.working_directory = Path(working_directory).absolute()
    self.answer_and_fix_issue_log_dir = self.working_directory / "answer_and_fix_issue_logs"
    self.answer_and_fix_issue_log_dir.mkdir(parents=True, exist_ok=True)

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
