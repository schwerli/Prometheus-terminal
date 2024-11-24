import logging

from langchain_core.language_models.chat_models import BaseChatModel

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.edit_node import EditNode
from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState
from prometheus.utils.issue_util import format_issue_comments


class BugFixingNode(EditNode):
  HUMAN_PROMPT = """\
You are a specialized code editing agent focused on fixing bugs. Your task is to analyze bug reports and implement precise fixes while preserving intended functionality.

TASK OVERVIEW:
Analyze the provided bug report and implement necessary fixes following these guidelines:
1. DO NOT modify test files - they define expected behavior
2. Make minimal, focused changes that fix the bug
3. Preserve existing functionality
4. Document your analysis and changes

THINKING PROCESS:
For each bug fix, follow these steps:
1. UNDERSTAND: Analyze the bug report and reproduction steps
2. DIAGNOSE: Identify the root cause of the issue
3. PLAN: Design a minimal fix that addresses the root cause
4. IMPLEMENT: Make precise code changes using edit tools
5. VERIFY: Ensure the fix addresses the reported issue

CURRENT CONTEXT:
Issue Title: {issue_title}
Description: {issue_body}
Comments: {issue_comments}

Bug Context:
{bug_context}

Previous Changes:
{previous_edit_info}

{reproduction_info}

OUTPUT FORMAT:
Structure your response as follows:
1. BUG ANALYSIS:
   - Root cause identification
   - Impact assessment
   - Potential fixes evaluation

2. IMPLEMENTATION PLAN:
   - Files to modify
   - Specific changes needed
   - Line numbers affected

3. CHANGES:
   - Detailed documentation of each change
   - Before/after code comparisons
   - Reasoning for changes

4. VERIFICATION:
   - How the fix addresses the issue
   - Potential side effects considered
   - Additional testing needed
"""

  def __init__(self, model: BaseChatModel, kg: KnowledgeGraph):
    super().__init__(model, kg)

    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_fixing_node")

  def format_human_message(self, state: IssueBugState):
    issue_comments = format_issue_comments(state["issue_comments"])

    previous_edit_info = ""
    if "patch" in state and state["patch"]:
      previous_edit_info = f"You have previously made the following changes:\n{state['patch']}"

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
