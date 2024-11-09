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
            "neither build nor test requested",
            {
                "run_build": False,
                "run_test": False,
                "build_success": False,
                "test_success": False
            },
            True
        ),
        (
            "only build requested and succeeds",
            {
                "run_build": True,
                "run_test": False,
                "build_success": True,
                "test_success": False
            },
            True
        ),
        (
            "only build requested and fails",
            {
                "run_build": True,
                "run_test": False,
                "build_success": False,
                "test_success": False
            },
            False
        ),
        (
            "only test requested and succeeds",
            {
                "run_build": False,
                "run_test": True,
                "build_success": False,
                "test_success": True
            },
            True
        ),
        (
            "only test requested and fails",
            {
                "run_build": False,
                "run_test": True,
                "build_success": False,
                "test_success": False
            },
            False
        ),
        (
            "both build and test requested and succeed",
            {
                "run_build": True,
                "run_test": True,
                "build_success": True,
                "test_success": True
            },
            True
        ),
        (
            "build fails but test succeeds",
            {
                "run_build": True,
                "run_test": True,
                "build_success": False,
                "test_success": True
            },
            False
        ),
        (
            "build succeeds but test fails",
            {
                "run_build": True,
                "run_test": True,
                "build_success": True,
                "test_success": False
            },
            False
        ),
        (
            "both build and test fail",
            {
                "run_build": True,
                "run_test": True,
                "build_success": False,
                "test_success": False
            },
            False
        ),
    ]
)
def test_router(router, test_case, state, expected):
    assert router(state) is expected, f"Failed case: {test_case}"