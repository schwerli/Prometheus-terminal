from prometheus.app.services.chat_service import ChatService
from prometheus.app.services.issue_service import IssueService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.configuration.config import settings


def get_services():
  neo4j_service = Neo4jService(settings.NEO4J_URI, settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
  postgres_service = PostgresService(settings.POSTGRES_URI)
  llm_service = LLMService(settings.LITELLM_MODEL, settings.LITELLM_ANTHROPIC_API_KEY)
  knowledge_graph_service = KnowledgeGraphService(neo4j_service, settings.NEO4J_BATCH_SIZE)
  resposistory_service = RepositoryService(
    knowledge_graph_service, settings.GITHUB_ACCESS_TOKEN, settings.MAX_AST_DEPTH
  )

  chat_service = ChatService(knowledge_graph_service, neo4j_service, postgres_service, llm_service)
  issue_service = IssueService(
    knowledge_graph_service, neo4j_service, postgres_service, llm_service
  )

  return (
    neo4j_service,
    postgres_service,
    knowledge_graph_service,
    resposistory_service,
    chat_service,
    issue_service,
  )
