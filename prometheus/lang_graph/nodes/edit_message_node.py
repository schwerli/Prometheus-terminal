import logging

from langchain_core.messages import HumanMessage

from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState
from prometheus.utils.issue_util import format_issue_info
from prometheus.utils.lang_graph_util import get_last_message_content


class EditMessageNode:
  FIRST_HUMAN_PROMPT = """\
{issue_info}

Bug Context:
{bug_context}

Bug analyzer agent has analyzed the issue and provided instruction on how to fix it:
{bug_analyzer_message}

Please implement these changes precisely, following the exact specifications from the analyzer.
"""

  FOLLOWUP_HUMAN_PROMPT = """\
The edit that you generated following error:
{edit_error}

{additional_context}

Bug analyzer agent has analyzed the issue and provided instruction on how to fix it:
{bug_analyzer_message}

Please implement these revised changes carefully, ensuring you address the
specific issues that caused the previous error.
"""

  def __init__(self):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.edit_message_node")

  def format_human_message(self, state: IssueBugState):
    edit_error = ""
    if "reproducing_test_fail_log" in state and state["reproducing_test_fail_log"]:
      edit_error = (
        f"Your failed to pass the bug exposing test cases:\n{state['reproducing_test_fail_log']}"
      )
    elif "build_fail_log" in state and state["build_fail_log"]:
      edit_error = f"Your failed to pass the build:\n{state['build_fail_log']}"
    elif "existing_test_fail_log" in state and state["existing_test_fail_log"]:
      edit_error = f"Your failed to existing test cases:\n{state['existing_test_fail_log']}"

    additional_context = ""
    if "refined_query" in state and state["refined_query"]:
      additional_context = "Additional context that might be useful:\n" + get_last_message_content(
        state["context_provider_messages"]
      )

    if edit_error:
      return HumanMessage(
        self.FOLLOWUP_HUMAN_PROMPT.format(
          edit_error=edit_error,
          additional_context=additional_context,
          bug_analyzer_message=get_last_message_content(state["issue_bug_analyzer_messages"]),
        )
      )

    return HumanMessage(
      self.FIRST_HUMAN_PROMPT.format(
        issue_info=format_issue_info(
          state["issue_title"], state["issue_body"], state["issue_comments"]
        ),
        bug_context=get_last_message_content(state["context_provider_messages"]),
        bug_analyzer_message=get_last_message_content(state["issue_bug_analyzer_messages"]),
      )
    )

  def __call__(self, state: IssueBugState):
    human_message = self.format_human_message(state)
    self._logger.debug(f"Sending message to EditNode:\n{human_message}")
    return {"edit_messages": [human_message]}
