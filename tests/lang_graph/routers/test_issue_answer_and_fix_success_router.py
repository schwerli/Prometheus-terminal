import pytest

from prometheus.lang_graph.routers.issue_answer_and_fix_success_router import (
  IssueAnswerAndFixSuccessRouter,
)


@pytest.fixture
def router():
  return IssueAnswerAndFixSuccessRouter()


@pytest.mark.parametrize(
  "test_case,state,expected",
  [
    (
      "neither build nor test requested - approved",
      {
        "reviewer_approved": True,
        "run_build": False,
        "run_test": False,
        "build_fail_log": "",
        "test_fail_log": "",
      },
      True,
    ),
    (
      "neither build nor test requested - not approved",
      {
        "reviewer_approved": False,
        "run_build": False,
        "run_test": False,
        "build_fail_log": "",
        "test_fail_log": "",
      },
      False,
    ),
    (
      "only build requested and succeeds - approved",
      {
        "reviewer_approved": True,
        "run_build": True,
        "run_test": False,
        "build_fail_log": "",
        "test_fail_log": "",
      },
      True,
    ),
    (
      "only build requested and succeeds - not approved",
      {
        "reviewer_approved": False,
        "run_build": True,
        "run_test": False,
        "build_fail_log": "",
        "test_fail_log": "",
      },
      False,
    ),
    (
      "only build requested and fails - approved",
      {
        "reviewer_approved": True,
        "run_build": True,
        "run_test": False,
        "build_fail_log": "Build failed",
        "test_fail_log": "",
      },
      False,
    ),
    (
      "only build requested and fails - not approved",
      {
        "reviewer_approved": False,
        "run_build": True,
        "run_test": False,
        "build_fail_log": "Build failed",
        "test_fail_log": "",
      },
      False,
    ),
    (
      "only test requested and succeeds - approved",
      {
        "reviewer_approved": True,
        "run_build": False,
        "run_test": True,
        "build_fail_log": "",
        "test_fail_log": "",
      },
      True,
    ),
    (
      "only test requested and succeeds - not approved",
      {
        "reviewer_approved": False,
        "run_build": False,
        "run_test": True,
        "build_fail_log": "",
        "test_fail_log": "",
      },
      False,
    ),
    (
      "only test requested and fails - approved",
      {
        "reviewer_approved": True,
        "run_build": False,
        "run_test": True,
        "build_fail_log": "",
        "test_fail_log": "Test failed",
      },
      False,
    ),
    (
      "only test requested and fails - not approved",
      {
        "reviewer_approved": False,
        "run_build": False,
        "run_test": True,
        "build_fail_log": "",
        "test_fail_log": "Test failed",
      },
      False,
    ),
    (
      "both build and test requested and succeed - approved",
      {
        "reviewer_approved": True,
        "run_build": True,
        "run_test": True,
        "build_fail_log": "",
        "test_fail_log": "",
      },
      True,
    ),
    (
      "both build and test requested and succeed - not approved",
      {
        "reviewer_approved": False,
        "run_build": True,
        "run_test": True,
        "build_fail_log": "",
        "test_fail_log": "",
      },
      False,
    ),
    (
      "build fails but test succeeds - approved",
      {
        "reviewer_approved": True,
        "run_build": True,
        "run_test": True,
        "build_fail_log": "Build failed",
        "test_fail_log": "",
      },
      False,
    ),
    (
      "build fails but test succeeds - not approved",
      {
        "reviewer_approved": False,
        "run_build": True,
        "run_test": True,
        "build_fail_log": "Build failed",
        "test_fail_log": "",
      },
      False,
    ),
    (
      "build succeeds but test fails - approved",
      {
        "reviewer_approved": True,
        "run_build": True,
        "run_test": True,
        "build_fail_log": "",
        "test_fail_log": "Test failed",
      },
      False,
    ),
    (
      "build succeeds but test fails - not approved",
      {
        "reviewer_approved": False,
        "run_build": True,
        "run_test": True,
        "build_fail_log": "",
        "test_fail_log": "Test failed",
      },
      False,
    ),
    (
      "both build and test fail - approved",
      {
        "reviewer_approved": True,
        "run_build": True,
        "run_test": True,
        "build_fail_log": "Build failed",
        "test_fail_log": "Test failed",
      },
      False,
    ),
    (
      "both build and test fail - not approved",
      {
        "reviewer_approved": False,
        "run_build": True,
        "run_test": True,
        "build_fail_log": "Build failed",
        "test_fail_log": "Test failed",
      },
      False,
    ),
  ],
)
def test_router(router, test_case, state, expected):
  assert router(state) is expected, f"Failed case: {test_case}"
