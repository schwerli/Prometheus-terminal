import logging

from langchain_core.messages import HumanMessage

from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState


class IssueBugContextFollowUpMessageNode:
  HUMAN_PROMPT = """\
I have generated this patch:
{patch}

But it resulted in this error:
{edit_error}

Now please think about what went wrong, and retrieve new context that may help me fix the bug.
"""

  def __init__(self):
    self._logger = logging.getLogger(
      "prometheus.lang_graph.nodes.issue_bug_context_follow_up_message_node"
    )

  def format_human_message(self, state: IssueBugState):
    edit_error = ""
    if "reproducing_test_fail_log" in state and state["reproducing_test_fail_log"]:
      edit_error = f"The patch failed to pass the bug exposing test cases:\n{state['reproducing_test_fail_log']}"
    elif "build_fail_log" in state and state["build_fail_log"]:
      edit_error = f"The patch failed to pass the build:\n{state['build_fail_log']}"
    elif "existing_test_fail_log" in state and state["existing_test_fail_log"]:
      edit_error = f"The patch failed to existing test cases:\n{state['existing_test_fail_log']}"

    return HumanMessage(
      self.HUMAN_PROMPT.format(
        patch=state["patch"],
        edit_error=edit_error,
      )
    )

  def __call__(self, state: IssueBugState):
    human_message = self.format_human_message(state)
    self._logger.debug(f"Sending query to context provider:\n{human_message}")
    return {"context_provider_messages": [human_message]}
