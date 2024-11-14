"""Router for handling issue answer and fix workflow when testing is needed.

This module provides a routing mechanism for the issue answer and fix process
when additional testing is required in the workflow.
"""

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueAnswerAndFixNeedTestRouter:
    """Router to determine if testing is required in the issue answer and fix workflow.

    This router checks the state to determine whether testing should be run
    as part of the issue resolution process.

    Attributes:
        None
    """

    def __call__(self, state: IssueAnswerAndFixState):
        """Determine if testing should be run based on the current workflow state.

        Args:
            state (IssueAnswerAndFixState): The current state of the issue 
                answer and fix workflow.

        Returns:
            bool: True if tests should be run, False otherwise.
        """
        return state["run_test"]
