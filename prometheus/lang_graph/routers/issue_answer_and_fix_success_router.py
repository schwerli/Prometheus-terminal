"""Router for determining success status of issue answer and fix operations.

This module provides a router that evaluates whether an issue answer and fix operation
was successful based on build and test results. It considers the operation successful if:
- Neither build nor test was requested
- Build was requested and completed without failures
- Test was requested and completed without failures
- Both build and test were requested and completed without failures
"""

import logging

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueAnswerAndFixSuccessRouter:
  """Routes based on the success status of issue answer and fix operations.

  This router evaluates the success of operations by checking build and test results
  in the provided state. It considers various combinations of build and test requests
  and their outcomes to determine if the overall operation was successful.
  """

  def __init__(self):
    """Initializes the router with a configured logger."""
    self._logger = logging.getLogger(
      "prometheus.lang_graph.routers.issue_answer_and_fix_success_router"
    )

  def __call__(self, state: IssueAnswerAndFixState) -> bool:
    """Determines if the issue answer and fix operation was successful.

    Args:
      state: A state object containing build and test execution results.
        Must include 'run_build', 'run_test', 'build_fail_log', and
        'test_fail_log' keys.

    Returns:
      bool: True if the operation was successful based on the requested
        operations and their results, False otherwise.
    """
    if not state["reviewer_approved"]:
      return False

    if not state["run_build"] and not state["run_test"]:
      return True

    if state["run_build"] and state["build_fail_log"]:
      return False

    if state["run_test"] and state["test_fail_log"]:
      return False

    return True
