"""Initializes and configures all prometheus services."""

from pathlib import Path

from prometheus.app.services.issue_service import IssueService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.app.services.service_coordinator import ServiceCoordinator
from prometheus.configuration.config import settings


def initialize_services() -> ServiceCoordinator:
  """Initializes and configures the complete prometheus service stack.

  This function creates and configures all required services for prometheus
  operation, using settings from the configuration module. It ensures proper
  initialization order and service dependencies.

  Note:
      This function assumes all required settings are properly configured in
      the settings module using Dynaconf. The following settings are required:
      - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
      - POSTGRES_URI
      - LITELLM_MODEL
      - NEO4J_BATCH_SIZE
      - KNOWLEDGE_GRAPH_MAX_AST_DEPTH
      - WORKING_DIRECTORY
      - GITHUB_ACCESS_TOKEN

  Returns:
      A fully configured ServiceCoordinator instance managing all services.
  """
  neo4j_service = Neo4jService(settings.NEO4J_URI, settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
  postgres_service = PostgresService(settings.POSTGRES_URI)
  llm_service = LLMService(
    settings.MODEL,
    getattr(settings, "OPENAI_API_KEY", None),
    getattr(settings, "ANTHROPIC_API_KEY", None),
    getattr(settings, "GEMINI_API_KEY", None),
  )
  knowledge_graph_service = KnowledgeGraphService(
    neo4j_service,
    settings.NEO4J_BATCH_SIZE,
    settings.KNOWLEDGE_GRAPH_MAX_AST_DEPTH,
    settings.KNOWLEDGE_GRAPH_CHUNK_SIZE,
    settings.KNOWLEDGE_GRAPH_CHUNK_OVERLAP,
  )
  resposistory_service = RepositoryService(knowledge_graph_service, settings.WORKING_DIRECTORY)
  issue_service = IssueService(
    knowledge_graph_service,
    resposistory_service,
    neo4j_service,
    postgres_service,
    llm_service,
    settings.MAX_TOKEN_PER_NEO4J_RESULT,
  )

  service_coordinator = ServiceCoordinator(
    issue_service,
    knowledge_graph_service,
    llm_service,
    neo4j_service,
    postgres_service,
    resposistory_service,
    settings.MAX_TOKEN_PER_NEO4J_RESULT,
    settings.GITHUB_ACCESS_TOKEN,
    Path(settings.WORKING_DIRECTORY).absolute(),
  )

  return service_coordinator
