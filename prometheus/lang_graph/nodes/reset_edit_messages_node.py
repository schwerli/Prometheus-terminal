

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class ResetEditMessagesNode:
    def __call__(self, state: IssueAnswerAndFixState):
        state["code_edit_messages"].clear()