from typing import TypedDict


class TestedPatchResult(TypedDict):
    # patch that was tested
    patch: str
    # whether the patch passed the regression tests
    passed: bool
    # regression test failure log if the patch did not pass, empty string otherwise
    regression_test_failure_log: str
