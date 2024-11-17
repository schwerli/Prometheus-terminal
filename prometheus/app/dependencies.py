"""Initializes and configures all prometheus services."""

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
  llm_service = LLMService(settings.LITELLM_MODEL)
  knowledge_graph_service = KnowledgeGraphService(
    neo4j_service,
    settings.NEO4J_BATCH_SIZE,
    settings.KNOWLEDGE_GRAPH_MAX_AST_DEPTH,
    settings.KNOWLEDGE_GRAPH_CHUNK_SIZE,
    settings.KNOWLEDGE_GRAPH_CHUNK_OVERLAP,
  )
  resposistory_service = RepositoryService(knowledge_graph_service, settings.WORKING_DIRECTORY)

  service_coordinator = ServiceCoordinator(
    knowledge_graph_service,
    llm_service,
    neo4j_service,
    postgres_service,
    resposistory_service,
    settings.GITHUB_ACCESS_TOKEN,
  )

  return service_coordinator
