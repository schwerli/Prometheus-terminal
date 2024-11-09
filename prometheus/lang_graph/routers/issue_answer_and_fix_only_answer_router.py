from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueAnswerAndFixOnlyAnswerRouter:
  def __call__(self, state: IssueAnswerAndFixState):
    return state["only_answer"]
