import uuid
from typing import Mapping, Optional, Sequence

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_subgraph import IssueAnswerAndFixSubgraph


class IssueAnswerAndFixService:
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
    self._initialize_issue_answer_and_fix_subgraph()

  def _initialize_issue_answer_and_fix_subgraph(self):
    if self.kg_service.kg is not None:
      self.issue_answer_and_fix_subgraph = IssueAnswerAndFixSubgraph(
        self.model,
        self.kg_service.kg,
        self.neo4j_service.neo4j_driver,
        self.postgres_service.checkpointer,
      )
    else:
      self.issue_answer_and_fix_subgraph = None

  def answer_and_fix_issue(
    self,
    project_path: str,
    title: str,
    body: str,
    comments: Sequence[Mapping[str, str]],
    thread_id: Optional[str] = None,
  ) -> str:
    if not self.issue_answer_and_fix_subgraph:
      raise ValueError("Knowledge graph not initialized")
    thread_id = str(uuid.uuid4()) if thread_id is None else thread_id
    return self.issue_answer_and_fix_subgraph.invoke(project_path, title, body, comments, thread_id)
