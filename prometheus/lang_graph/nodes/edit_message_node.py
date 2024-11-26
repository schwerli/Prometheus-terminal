import logging

from langchain_core.messages import HumanMessage

from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState
from prometheus.utils.issue_util import format_issue_info


class EditMessageNode:
  FIRST_HUMAN_PROMPT = """\
You are a specialized bug fixing agent focused on analyzing bug reports and implementing precise fixes while preserving intended functionality. You inherit precise code editing capabilities from your parent class.

CRITICAL BUG FIXING PRINCIPLES:
1. DO NOT modify test files - they define expected behavior
2. Each bug fix should target exactly one issue
3. Preserve all existing correct functionality
4. Changes must not introduce new bugs

BUG FIXING PROCESS:
1. UNDERSTAND THE BUG
   - Read the bug reproducing file if it is provided
   - Identify expected vs actual behavior
   - Consider edge cases that might be affected

2. ROOT CAUSE ANALYSIS
   - Trace through the code execution path
   - Identify where behavior diverges from expected
   - Look for related code that might be affected

3. FIX DESIGN
   - Consider multiple potential solutions
   - Choose the most robust and minimal fix
   - Ensure fix handles all edge cases
   - Verify fix preserves existing behavior

{issue_info}

Bug Context:
{bug_context}
"""

  FOLLOWUP_HUMAN_PROMPT = """\
The edit that you generated following error:
{edit_error}

I have also retrived additional context from codebase that may help you fix the bug:
{additional_context}

Now think about what went wrong and try to fix the bug again.
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

    if edit_error:
      return HumanMessage(
        self.FOLLOWUP_HUMAN_PROMPT.format(
          edit_error=edit_error,
          additional_context=state["context_provider_messages"][-1].content,
        )
      )

    return HumanMessage(
      self.FIRST_HUMAN_PROMPT.format(
        issue_info=format_issue_info(
          state["issue_title"], state["issue_body"], state["issue_comments"]
        ),
        bug_context=state["context_provider_messages"][-1].content,
      )
    )

  def __call__(self, state: IssueBugState):
    human_message = self.format_human_message(state)
    self._logger.debug(f"Sending message to EditNode:\n{human_message}")
    return {"edit_messages": [human_message]}
