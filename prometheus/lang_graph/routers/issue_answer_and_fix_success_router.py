from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueAnswerAndFixSuccessRouter:
  def __call__(self, state: IssueAnswerAndFixState):
    if not state["run_build"] and not state["run_test"]:
      return True

    if state["run_build"] and not state["build_success"]:
      return False

    if state["run_test"] and not state["test_success"]:
      return False

    return state["build_success"] and state["test_success"]
