from typing import Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueState
from prometheus.lang_graph.subgraphs.issue_bug_subgraph import IssueBugSubgraph


class IssueBugSubgraphNode:
  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.issue_bug_subgraph = IssueBugSubgraph(
      model,
      container,
      kg,
      neo4j_driver,
      max_token_per_neo4j_result,
      build_commands,
      test_commands,
      thread_id,
      checkpointer,
    )

  def __call__(self, state: IssueState):
    output_state = self.issue_bug_subgraph.invoke(
      state["issue_title"],
      state["issue_body"],
      state["issue_comments"],
      state["run_build"],
      state["run_existing_test"],
    )
    return {
      "issue_response": output_state["issue_response"],
      "patch": output_state["edit_patch"],
      "reproduced_bug_file": output_state["reproduced_bug_file"],
    }
