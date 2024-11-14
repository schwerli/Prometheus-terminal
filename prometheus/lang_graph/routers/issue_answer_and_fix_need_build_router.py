"""Router that determines if a build step is needed in the issue answer and fix workflow.

This module provides a router class that checks the state of an issue answer and fix
workflow to determine if a build step should be executed.
"""

import logging

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueAnswerAndFixNeedBuildRouter:
  """Router that controls the execution of build steps in the workflow.

  This router examines the state of an issue answer and fix workflow to determine
  if a build step should be executed based on the 'run_build' flag in the state.
  """

  def __init__(self):
    """Initializes the router with a configured logger."""
    self._logger = logging.getLogger(
      "prometheus.lang_graph.routers.issue_answer_and_fix_need_build_router"
    )

  def __call__(self, state: IssueAnswerAndFixState) -> bool:
    """Determine if a build step should be executed.

    Args:
        state (IssueAnswerAndFixState): The current state of the issue answer and fix workflow.

    Returns:
        bool: True if a build should be executed, False otherwise.
    """
    self._logger.info(f"Building required? {state['run_build']}")
    return state["run_build"]
