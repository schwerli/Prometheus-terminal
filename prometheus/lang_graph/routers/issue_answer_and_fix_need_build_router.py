from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueAnswerAndFixNeedBuildRouter:
  def __call__(self, state: IssueAnswerAndFixState):
    return state["run_build"]
    
