"""Initializes and configures all prometheus services."""

from prometheus.app.services.base_service import BaseService
from prometheus.app.services.issue_service import IssueService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.configuration.config import settings


def initialize_services() -> dict[str, BaseService]:
    """Initializes and configures the complete prometheus service stack.

    This function creates and configures all required services for prometheus
    operation, using settings from the configuration module. It ensures proper
    initialization order and service dependencies.

    Note:
        This function assumes all required settings are properly configured in
        the settings module using Dynaconf. The following settings are required:
        - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
        - LITELLM_MODEL
        - NEO4J_BATCH_SIZE
        - KNOWLEDGE_GRAPH_MAX_AST_DEPTH
        - WORKING_DIRECTORY
        - GITHUB_ACCESS_TOKEN

    Returns:
        A fully configured ServiceCoordinator instance managing all services.
    """
    neo4j_service = Neo4jService(
        settings.NEO4J_URI, settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD
    )
    llm_service = LLMService(
        settings.ADVANCED_MODEL,
        settings.BASE_MODEL,
        getattr(settings, "OPENAI_FORMAT_API_KEY", None),
        getattr(settings, "OPENAI_FORMAT_BASE_URL", None),
        getattr(settings, "ANTHROPIC_API_KEY", None),
        getattr(settings, "GEMINI_API_KEY", None),
        getattr(settings, "TEMPERATURE", None),
        getattr(settings, "MAX_OUTPUT_TOKENS", None),
    )
    knowledge_graph_service = KnowledgeGraphService(
        neo4j_service,
        settings.NEO4J_BATCH_SIZE,
        settings.KNOWLEDGE_GRAPH_MAX_AST_DEPTH,
        settings.KNOWLEDGE_GRAPH_CHUNK_SIZE,
        settings.KNOWLEDGE_GRAPH_CHUNK_OVERLAP,
    )
    repository_service = RepositoryService(knowledge_graph_service, settings.WORKING_DIRECTORY)
    issue_service = IssueService(
        knowledge_graph_service,
        repository_service,
        neo4j_service,
        llm_service,
        settings.MAX_TOKEN_PER_NEO4J_RESULT,
        settings.WORKING_DIRECTORY,
    )

    return {
        "neo4j_service": neo4j_service,
        "llm_service": llm_service,
        "knowledge_graph_service": knowledge_graph_service,
        "repository_service": repository_service,
        "issue_service": issue_service,
    }
