import logging
from typing import Dict

from langchain_core.messages import HumanMessage

from prometheus.utils.issue_util import format_issue_info


class IssueBugContextMessageNode:
  HUMAN_PROMPT = (
    """\
{issue_info}

Find relevant source code context needed to understand and fix this issue.
Retrieve complete source files or classes where the issue might occur.

Follow these steps:
1. Identify key components mentioned in the issue (functions, classes, types, etc.)
2. Find their complete implementations and class definitions
3. Include related code from the same module that affects the behavior
4. Follow imports to find dependent code that directly impacts the issue

You should ignore test files
""".replace("{", "{{")
    .replace("}", "}}")
    .replace("{{issue_info}}", "{issue_info}")
  )

  def __init__(self):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.issue_bug_context_message_node")

  def __call__(self, state: Dict):
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(
        issue_info=format_issue_info(
          state["issue_title"], state["issue_body"], state["issue_comments"]
        ),
      )
    )
    self._logger.debug(f"Sending query to ContextProviderNode:\n{human_message}")
    return {"context_provider_messages": [human_message]}
