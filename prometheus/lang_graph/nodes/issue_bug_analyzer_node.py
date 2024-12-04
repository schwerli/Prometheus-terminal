import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState
from prometheus.utils.lang_graph_util import truncate_messages


class IssueBugAnalyzerNode:
  SYS_PROMPT = """\
You are an expert software engineer specializing in bug analysis and fixes. Your role is to:

1. Carefully analyze reported software issues and bugs
2. Trace code execution paths to identify where things go wrong
3. Determine root causes through systematic investigation
4. Propose precise, practical fixes

When suggesting fixes:
- Specify exact file paths, line numbers, and code changes
- Write code that matches the project's style and conventions
- Consider the broader system impact of any changes
- Ensure changes are minimal and focused on the specific bug

For patch failures:
- Analyze error messages and build failures carefully
- Identify specific issues in the failed patch
- Propose revised solutions that address the root cause while avoiding the previous errors

Keep responses clear, technical, and focused on the specific issue at hand. Your suggestions will
be used by an automated edit agent to implement the changes, so be precise in specifying code modifications.

Communicate in a professional, direct manner focused on technical accuracy and practical solutions.
"""

  def __init__(self, model: BaseChatModel):
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.model = model
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.issue_bug_analyzer_node")

  def __call__(self, state: IssueBugState):
    message_history = [self.system_prompt] + state["issue_bug_analyzer_messages"]
    truncated_message_history = truncate_messages(message_history)
    response = self.model.invoke(truncated_message_history)

    self._logger.debug(f"IssueBugAnalyzerNode response:\n{response}")
    return {"issue_bug_analyzer_messages": [response]}
