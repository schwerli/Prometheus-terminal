import uuid
from pathlib import Path
from typing import Mapping, Optional, Sequence

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.docker.general_container import GeneralContainer
from prometheus.docker.user_defined_container import UserDefinedContainer
from prometheus.lang_graph.graphs.issue_graph import IssueGraph
from prometheus.lang_graph.graphs.issue_state import IssueType


class IssueService:
  def __init__(
    self,
    kg_service: KnowledgeGraphService,
    neo4j_service: Neo4jService,
    postgres_service: PostgresService,
    llm_service: LLMService,
    max_token_per_neo4j_result: int,
  ):
    self.kg_service = kg_service
    self.neo4j_service = neo4j_service
    self.postgres_service = postgres_service
    self.llm_service = llm_service
    self.max_token_per_neo4j_result = max_token_per_neo4j_result

  def answer_issue(
    self,
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
  ):
    if dockerfile_content or image_name:
      container = UserDefinedContainer(
        Path(self.kg_service.kg.get_local_path()),
        build_commands,
        test_commands,
        workdir,
        dockerfile_content,
        image_name,
      )
    else:
      container = GeneralContainer(Path(self.kg_service.kg.get_local_path()))

    thread_id = str(uuid.uuid4())
    issue_graph = IssueGraph(
      self.llm_service.model,
      self.kg_service.kg,
      self.neo4j_service.neo4j_driver,
      self.max_token_per_neo4j_result,
      container,
      build_commands,
      test_commands,
      thread_id,
      self.postgres_service.checkpointer,
    )

    issue_graph.invoke(
      issue_title, issue_body, issue_comments, issue_type, run_build, run_existing_test
    )
