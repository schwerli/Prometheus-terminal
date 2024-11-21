import logging
from typing import Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueState
from prometheus.lang_graph.subgraphs.bug_reproduction_subgraph import BugReproductionSubgraph


class BugReproductionSubgraphNode:
  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    kg: KnowledgeGraph,
    test_commands: Optional[Sequence[str]],
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_reproduction_subgraph_node")
    self.bug_reproduction_subgraph = BugReproductionSubgraph(
      model, container, kg, test_commands, thread_id, checkpointer
    )

  def __call__(self, state: IssueState):
    self._logger.info("Enter bug_reproduction_subgraph")
    self._logger.debug(f"issue_title: {state['issue_title']}")
    self._logger.debug(f"issue_body: {state['issue_body']}")
    self._logger.debug(f"issue_comments: {state['issue_comments']}")
    self._logger.debug(f"bug_context: {state['bug_context']}")

    try:
      output_state = self.bug_reproduction_subgraph.invoke(
        state["issue_title"],
        state["issue_body"],
        state["issue_comments"],
        state["bug_context"],
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
