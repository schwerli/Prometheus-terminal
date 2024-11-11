import logging

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueAnswerAndFixSuccessRouter:
  def __init__(self):
    self._logger = logging.getLogger(
      "prometheus.lang_graph.routers.issue_answer_and_fix_success_router"
    )

  def __call__(self, state: IssueAnswerAndFixState):
    if not state["run_build"] and not state["run_test"]:
      return True

    if state["run_build"] and state["build_fail_log"]:
      return False

    if state["run_test"] and state["test_fail_log"]:
      return False

    return True
