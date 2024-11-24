import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from prometheus.docker.base_container import BaseContainer
from prometheus.lang_graph.subgraphs.bug_fix_verification_subgraph import BugFixVerificationSubgraph
from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState


class BugFixVerificationSubgraphNode:
  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self._logger = logging.getLogger(
      "prometheus.lang_graph.nodes.bug_fix_verification_subgraph_node"
    )
    self.subgraph = BugFixVerificationSubgraph(model, container, thread_id, checkpointer)

  def __call__(self, state: IssueBugState):
    self._logger.info("Enter bug_fix_verification_subgraph_node")
    self._logger.debug(f"reproduced_bug_file: {state['reproduced_bug_file']}")
    self._logger.debug(f"reproduced_bug_commands: {state['reproduced_bug_commands']}")

    output_state = self.subgraph.invoke(
      state["reproduced_bug_file"], state["reproduced_bug_commands"]
    )

    self._logger.info(f"reproducing_test_passed: {output_state['reproducing_test_passed']}")
    self._logger.info(f"reproducing_test_fail_log: {output_state['reproducing_test_fail_log']}")

    return {
      "reproducing_test_passed": output_state["reproducing_test_passed"],
      "reproducing_test_fail_log": output_state["reproducing_test_fail_log"],
    }
