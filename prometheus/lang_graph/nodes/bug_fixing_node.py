import logging

from langchain_core.language_models.chat_models import BaseChatModel

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.edit_node import EditNode
from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState
from prometheus.utils.issue_util import format_issue_comments


class BugFixingNode(EditNode):
  HUMAN_PROMPT = """\
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

CURRENT CONTEXT:
Issue Title: {issue_title}
Description: {issue_body}
Comments: {issue_comments}

Bug Context:
{bug_context}

Previous Changes:
{previous_edit_info}

{reproduction_info}
"""

  def __init__(self, model: BaseChatModel, kg: KnowledgeGraph):
    super().__init__(model, kg)

    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_fixing_node")

  def format_human_message(self, state: IssueBugState):
    issue_comments = format_issue_comments(state["issue_comments"])

    previous_edit_info = ""
    if "patch" in state and state["patch"]:
      previous_edit_info = f"You have previously made the following changes:\n{state['patch']}"

    reproduction_info = ""
    if state["reproduced_bug"]:
      reproduction_info = (
        f"BUG REPRODUCTION:\nThe bug has been reproduced in file (please read it): {state['reproduced_bug_file']}"
      )

    return self.HUMAN_PROMPT.format(
      issue_title=state["issue_title"],
      issue_body=state["issue_body"],
      issue_comments=issue_comments,
      bug_context=state["bug_context"],
      previous_edit_info=previous_edit_info,
      reproduction_info=reproduction_info,
    )

  def __call__(self, state: IssueBugState):
    human_message = self.format_human_message(state)
    message_history = [self.system_prompt, human_message] + state["bug_fixing_messages"]

    response = self.model_with_tool.invoke(message_history)
    self._logger.debug(f"BugFixingNode response:\n{response}")
    return {"bug_fixing_messages": [response]}
