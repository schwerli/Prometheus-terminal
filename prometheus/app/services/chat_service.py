import uuid
from typing import Optional

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.lang_graph.subgraphs import context_provider_subgraph


class ChatService:
  def __init__(
    self,
    kg_service: KnowledgeGraphService,
    neo4j_service: Neo4jService,
    postgres_service: PostgresService,
    llm_serivice: LLMService,
  ):
    self.kg_service = kg_service
    self.neo4j_service = neo4j_service
    self.postgres_service = postgres_service
    self.model = llm_serivice.model
    self._initialize_chat_subgraph()

  def _initialize_chat_subgraph(self):
    if self.kg_service.kg is not None:
      self.cp_subgraph = context_provider_subgraph.ContextProviderSubgraph(
        self.model,
        self.kg_service.kg,
        self.neo4j_service.neo4j_driver,
        self.postgres_service.checkpointer,
      )
    else:
      self.cp_subgraph = None

  def chat(self, query: str, thread_id: Optional[str] = None) -> tuple[str, str]:
    thread_id = str(uuid.uuid4()) if thread_id is None else thread_id
    if not self.cp_subgraph:
      raise ValueError("Knowledge graph not initialized")
    response = self.cp_subgraph.invoke(query, thread_id)
    return thread_id, response
