"""Service for handling issue analysis, answering, and fix generation.

This service is connecting the received issue from API request to the IssueAnswerAndFixSubgraph.
The main logic is in IssueAnswerAndFixSubgraph.py.
"""

import uuid
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.postgres_service import PostgresService
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_subgraph import IssueAnswerAndFixSubgraph


class IssueAnswerAndFixService:
  """Service for initializing and calling IssueAnswerAndFixSubgraph."""

  def __init__(
    self,
    kg_service: KnowledgeGraphService,
    neo4j_service: Neo4jService,
    postgres_service: PostgresService,
    llm_serivice: LLMService,
    local_path: Path,
    dockerfile_content: Optional[str] = None,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
  ):
    """
    Args:
      kg_service: Knowledge graph service (For creating the knowledge graph).
      neo4j_service: Neo4j database service (For persistance of knowledge graph).
      postgres_service: PostgreSQL database service (For persistance of agent state).
      llm_service: LLM service (For creating the LLM model).
      local_path: Path to local repository.
      dockerfile_content: Optional user defined Dockerfile content for the containerized enviroment.
      build_commands: Optional build commands for the containerized enviroment.
      test_commands: Optional test commands for the containerized enviroment.
    """
    if kg_service.kg is None:
      raise ValueError("Knowledge graph not initialized when initializing IssueAnswerAndFixService")

    self.issue_answer_and_fix_subgraph = IssueAnswerAndFixSubgraph(
      llm_serivice.model,
      kg_service.kg,
      neo4j_service.neo4j_driver,
      local_path,
      postgres_service.checkpointer,
      dockerfile_content,
      build_commands,
      test_commands,
    )

  def answer_and_fix_issue(
    self,
    title: str,
    body: str,
    comments: Sequence[Mapping[str, str]],
    only_answer: bool,
    run_build: bool,
    run_tests: bool,
    thread_id: Optional[str] = None,
  ) -> Tuple[str, str]:
    """Calls the IssueAnswerAndFixSubgraph to answer and fix the issue.

    Args:
      title: The title of the issue.
      body: The main description of the issue.
      comments: Sequence of comment dictionaries related to the issue.
      only_answer: If True, only generates an answer without implementing a fix.
      run_build: If True, runs build validation on any generated fix.
      run_tests: If True, runs tests on any generated fix.
      thread_id: Optional identifier for the processing thread. If None,
          a new UUID will be generated.

    Returns:
      A tuple containing (issue_response, patch), where:
        - issue_response is a string containing the generated response
        - patch is the generated fix if only_answer is False, otherwise it is an empty string
    """
    thread_id = str(uuid.uuid4()) if thread_id is None else thread_id
    return self.issue_answer_and_fix_subgraph.invoke(
      title, body, comments, only_answer, run_build, run_tests, thread_id
    )
