from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class ResetMessagesNode:
  def __init__(self, message_state_key: str):
    self.message_state_key = message_state_key

  def __call__(self, state: IssueAnswerAndFixState):
    state[self.message_state_key].clear()
