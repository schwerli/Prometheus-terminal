import logging
from typing import Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_reproduction_subgraph import BugReproductionSubgraph
from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState


class BugReproductionSubgraphNode:
  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    test_commands: Optional[Sequence[str]],
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_reproduction_subgraph_node")
    self.bug_reproduction_subgraph = BugReproductionSubgraph(
      model,
      container,
      kg,
      neo4j_driver,
      max_token_per_neo4j_result,
      test_commands,
      thread_id,
      checkpointer,
    )

  def __call__(self, state: IssueBugState):
    self._logger.info("Enter bug_reproduction_subgraph")
    self._logger.debug(f"issue_title: {state['issue_title']}")
    self._logger.debug(f"issue_body: {state['issue_body']}")
    self._logger.debug(f"issue_comments: {state['issue_comments']}")

    try:
      output_state = self.bug_reproduction_subgraph.invoke(
        state["issue_title"],
        state["issue_body"],
        state["issue_comments"],
      )
    except GraphRecursionError:
      self._logger.info("Recursion limit reached, returning reproduced_bug=False")
      return {"reproduced_bug": False}

    self._logger.info(f"reproduced_bug: {output_state['reproduced_bug']}")
    self._logger.info(f"reproduced_bug_file: {output_state['reproduced_bug_file']}")
    self._logger.info(f"reproduced_bug_commands: {output_state['reproduced_bug_commands']}")
    return {
      "reproduced_bug": output_state["reproduced_bug"],
      "reproduced_bug_file": output_state["reproduced_bug_file"],
      "reproduced_bug_commands": output_state["reproduced_bug_commands"],
    }
