import logging
from typing import Dict, Optional

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.issue_not_verified_bug_subgraph import (
  IssueNotVerifiedBugSubgraph,
)


class IssueNotVerifiedBugSubgraphNode:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    git_repo: GitRepository,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self._logger = logging.getLogger(
      "prometheus.lang_graph.nodes.issue_not_verified_bug_subgraph_node"
    )
    self.issue_not_verified_bug_subgraph = IssueNotVerifiedBugSubgraph(
      model=model,
      kg=kg,
      git_repo=git_repo,
      neo4j_driver=neo4j_driver,
      max_token_per_neo4j_result=max_token_per_neo4j_result,
      thread_id=thread_id,
      checkpointer=checkpointer,
    )

  def __call__(self, state: Dict):
    self._logger.info("Enter IssueNotVerifiedBugSubgraphNode")

    output_state = self.issue_not_verified_bug_subgraph.invoke(
      issue_title=state["issue_title"],
      issue_body=state["issue_body"],
      issue_comments=state["issue_comments"],
      number_of_candidate_patch=state["number_of_candidate_patch"],
    )

    self._logger.info(f"final_patch:\n{output_state['final_patch']}")

    return {
      "edit_patch": output_state["final_patch"],
      "passed_reproducing_test": False,
      "passed_build": False,
      "passed_existing_test": False,
    }
