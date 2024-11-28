import logging

from langchain_core.messages import HumanMessage

from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState


class BugReproducingContextFollowUpMessageNode:
  HUMAN_PROMPT = """\
I have generated the following self-contained bug exposing test case:
{bug_exposing_test}

But it failed to reproduce the bug because:
{reproduced_bug_failure_log}

Now please think about what went wrong, and retrieve new context that may help me reproduce the bug.
"""

  def __init__(self):
    self._logger = logging.getLogger(
      "prometheus.lang_graph.nodes.bug_reproducing_context_follow_up_message_node"
    )

  def format_human_message(self, state: BugReproductionState):
    return HumanMessage(
      self.HUMAN_PROMPT.format(
        bug_exposing_test=state["bug_reproducing_write_messages"][-1].content,
        reproduced_bug_failure_log=state["reproduced_bug_failure_log"],
      )
    )

  def __call__(self, state: BugReproductionState):
    human_message = self.format_human_message(state)
    self._logger.debug(f"Sending query to context provider:\n{human_message}")
    return {"context_provider_messages": [human_message]}
