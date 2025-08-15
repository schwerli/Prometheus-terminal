from operator import add
from typing import Annotated, Mapping, Sequence, TypedDict

from prometheus.models.context import Context


class BugRegressionState(TypedDict):
    issue_title: str
    issue_body: str
    issue_comments: Sequence[Mapping[str, str]]

    max_refined_query_loop: int

    select_regression_query: str
    select_regression_context: Sequence[Context]

    # Number of regression tests to select
    number_of_selected_regression_tests: int
    # Selected regression tests to run
    selected_regression_tests: Sequence[str]

    # Passed regression tests before and after the fix
    before_passed_regression_tests: Sequence[str]
    after_passed_regression_tests: Sequence[str]

    # Log of regression test failures of the last run
    regression_test_fail_log: str

    # List of patches generated to fix the issue
    untested_patches: Sequence[str]
    # List of patches that passed the regression tests
    passed_patches: Annotated[Sequence[str], add]

    # Current patch being tested
    current_patch: str
