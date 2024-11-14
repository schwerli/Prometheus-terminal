import logging

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import (
  IssueAnswerAndFixState,
  ResponseModeEnum,
)


class IssueAnswerAndFixOnlyAnswerRouter:
  def __init__(self):
    """Initializes the router with a configured logger."""
    self._logger = logging.getLogger(
      "prometheus.lang_graph.routers.issue_answer_and_fix_only_answer_router"
    )

  def __call__(self, state: IssueAnswerAndFixState):
    if state["response_mode"] == ResponseModeEnum.ONLY_ANSWER:
      self._logger.info("Only answer requested by the user")
      return True

    if "require_edit" not in state:
      self._logger.info("RequireEditClassifierNode not executed yet")
      return False

    self._logger.info(f"Edit required? {state['require_edit']}")
    return not state["require_edit"]
