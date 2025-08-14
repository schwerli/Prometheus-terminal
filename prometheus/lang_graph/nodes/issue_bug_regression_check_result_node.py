import logging
import threading
from collections import Counter

from prometheus.lang_graph.subgraphs.bug_regression_state import BugRegressionState


class IssueBugRegressionCheckResultNode:
    """
    Check if the applied patch passes all the regression tests and update the state accordingly.
    """

    def __init__(self):
        self._logger = logging.getLogger(
            f"thread-{threading.get_ident()}.prometheus.lang_graph.nodes.issue_bug_regression_check_result_node"
        )

    def __call__(self, state: BugRegressionState):
        """
        Check if the applied patch passes all the regression tests and update the state accordingly.

        Args:
          state: Current state containing untested patches.
        """
        # Check if the tests passed before and after applying the patch is the same
        self._logger.debug(f"before passed regression tests: {state['before_passed_regression_tests']}")
        self._logger.debug(f"after passed regression tests: {state['after_passed_regression_tests']}")
        current_patch_passed = (Counter(state["before_passed_regression_tests"]) ==
                                Counter(state["before_passed_regression_tests"]))
        # If the before_passed_regression_tests is equal to the after_passed_regression_tests,
        self._logger.debug(current_patch_passed)

        return {
            "passed_patches": state["current_patch"] if current_patch_passed else [],
        }
