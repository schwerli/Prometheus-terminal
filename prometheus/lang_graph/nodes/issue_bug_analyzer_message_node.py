import logging
from typing import Dict

from langchain_core.messages import HumanMessage

from prometheus.utils.issue_util import format_issue_info
from prometheus.utils.lang_graph_util import get_last_message_content


class IssueBugAnalyzerMessageNode:
  FIRST_HUMAN_PROMPT = """\
I am going to share details about a issue reported to a codebase and its related bug context. Please help analyze this bug by:

1. Understanding the Issue:
- Analyze the issue title, description, and any comments provided
- Identify the reported symptoms and unexpected behaviors

2. Code Analysis:
- Examine the provided source code files and documentation
- Trace the code execution path that leads to the bug
- Identify where and why the code deviates from expected behavior

3. Root Cause Analysis:
- Determine the fundamental cause of the bug
- Explain which specific code components or interactions are responsible
- Highlight any underlying design issues or assumptions that contributed to the bug

4. Bug Fix:
- Suggest specific code changes to fix the bug
- Explain why the proposed changes would resolve the issue
- Consider potential side effects on other system components

Here are the details for analysis:

{issue_info}

Bug Context:
{bug_context}
"""

  FOLLOWUP_HUMAN_PROMPT = """\
Given your suggestion, the edit agent generated the following patch:
{edit_patch}

The patch generated following error:
{edit_error}

{additional_context}

Please:
1. Analyze why the patch failed
2. Identify any issues with the original patch (syntax errors, incorrect file paths, context mismatches, etc.)
3. Provide a revised fix that addresses the error

Please provide your revised solution following the same detailed format as before.
"""

  def __init__(self):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.issue_bug_analyzer_message_node")

  def format_human_message(self, state: Dict):
    edit_error = ""
    if "reproducing_test_fail_log" in state and state["reproducing_test_fail_log"]:
      edit_error = f"The patch failed to pass the bug exposing test cases:\n{state['reproducing_test_fail_log']}"
    elif "build_fail_log" in state and state["build_fail_log"]:
      edit_error = f"The patch failed to pass the build:\n{state['build_fail_log']}"
    elif "existing_test_fail_log" in state and state["existing_test_fail_log"]:
      edit_error = f"The patch failed to existing test cases:\n{state['existing_test_fail_log']}"

    if not edit_error:
      return HumanMessage(
        self.FIRST_HUMAN_PROMPT.format(
          issue_info=format_issue_info(
            state["issue_title"], state["issue_body"], state["issue_comments"]
          ),
          bug_context=get_last_message_content(state["context_provider_messages"]),
        )
      )

    additional_context = ""
    if "refined_query" in state and state["refined_query"]:
      additional_context = "Additional context that might be useful:\n" + get_last_message_content(
        state["context_provider_messages"]
      )

    return HumanMessage(
      self.FOLLOWUP_HUMAN_PROMPT.format(
        edit_patch=state["edit_patch"],
        edit_error=edit_error,
        additional_context=additional_context,
      )
    )

  def __call__(self, state: Dict):
    human_message = self.format_human_message(state)
    self._logger.debug(f"Sending message to IssueBugAnalyzerNode:\n{human_message}")
    return {"issue_bug_analyzer_messages": [human_message]}
