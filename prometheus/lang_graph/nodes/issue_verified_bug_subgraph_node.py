import logging
from typing import Dict, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError

from prometheus.docker.base_container import BaseContainer
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.issue_verified_bug_subgraph import IssueVerifiedBugSubgraph


class IssueVerifiedBugSubgraphNode:
  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    kg: KnowledgeGraph,
    git_repo: GitRepository,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self._logger = logging.getLogger(
      "prometheus.lang_graph.nodes.issue_not_verified_bug_subgraph_node"
    )
    self.issue_reproduced_bug_subgraph = IssueVerifiedBugSubgraph(
      model=model,
      container=container,
      kg=kg,
      git_repo=git_repo,
      neo4j_driver=neo4j_driver,
      max_token_per_neo4j_result=max_token_per_neo4j_result,
      build_commands=build_commands,
      test_commands=test_commands,
      thread_id=thread_id,
      checkpointer=checkpointer,
    )

  def __call__(self, state: Dict):
    self._logger.info("Enter issue_verified_bug_subgraph_node")
    try:
      output_state = self.issue_reproduced_bug_subgraph.invoke(
        state["issue_title"],
        state["issue_body"],
        state["issue_comments"],
        state["run_build"],
        state["run_existing_test"],
        state["reproduced_bug_file"],
        state["reproduced_bug_commands"],
      )
    except GraphRecursionError:
      self._logger.info("Recursion limit reached")
      return {
        "edit_patch": "",
        "passed_reproducing_test": False,
        "passed_build": False,
        "passed_existing_test": False,
      }

    return {
      "edit_patch": output_state["edit_patch"],
      "passed_reproducing_test": not bool(output_state["reproducing_test_fail_log"]),
      "passed_build": state["run_build"] and not output_state["build_fail_log"],
      "passed_existing_test": state["run_existing_test"]
      and not output_state["existing_test_fail_log"],
    }
