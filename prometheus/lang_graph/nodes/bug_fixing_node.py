import logging

from langchain_core.language_models.chat_models import BaseChatModel

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.edit_node import EditNode
from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState
from prometheus.utils.issue_util import format_issue_comments


class BugFixingNode(EditNode):
  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {issue_title}
Description: {issue_body}
Comments: {issue_comments}

BUG CONTEXT:
{bug_context}

{reproduction_info}

{previous_edit_info}

YOUR TASK:
You are tasked with fixing this bug by making precise code changes. Important guidelines:
1. DO NOT modify any test files - tests are the source of truth for expected behavior
2. Make minimal, focused changes to fix the bug while preserving existing functionality
3. For each change:
   - Read current file state
   - Plan precise changes
   - Make the change
   - Verify immediately

Please proceed with fixing the bug, documenting your analysis and changes.
"""

  def __init__(self, model: BaseChatModel, kg: KnowledgeGraph):
    super().__init__(model, kg)

    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_fixing_node")

  def format_human_message(self, state: IssueBugState):
    issue_comments = format_issue_comments(state["issue_comments"])

    reproduction_info = ""
    if state["reproduced_bug"]:
      reproduction_info = (
        f"BUG REPRODUCTION:\nThe bug has been reproduced in file: {state['reproduced_bug_file']}"
      )

    previous_edit_info = ""
    if state["patch"]:
      previous_edit_info = f"You have previously made the following changes:\n{state['patch']}"

    return self.HUMAN_PROMPT.format(
      issue_title=state["issue_title"],
      issue_body=state["issue_body"],
      issue_comments=issue_comments,
      bug_context=state["bug_context"],
      reproduction_info=reproduction_info,
      previous_edit_info=previous_edit_info,
    )

  def __call__(self, state: IssueBugState):
    human_message = self.format_human_message(state)
    message_history = [self.system_prompt, human_message] + state["bug_fixing_messages"]

    response = self.model_with_tool.invoke(message_history)
    self._logger.debug(f"BugFixingNode response:\n{response}")
    return {"bug_fixing_messages": [response]}
