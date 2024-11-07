from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.app.services.service_coordinator import ServiceCoordinator
from prometheus.configuration.config import settings


def initialize_services() -> ServiceCoordinator:
  neo4j_service = Neo4jService(settings.NEO4J_URI, settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
  postgres_service = PostgresService(settings.POSTGRES_URI)
  llm_service = LLMService(settings.LITELLM_MODEL)
  knowledge_graph_service = KnowledgeGraphService(
    neo4j_service, settings.NEO4J_BATCH_SIZE, settings.KNOWLEDGE_GRAPH_MAX_AST_DEPTH
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
