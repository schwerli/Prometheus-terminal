from prometheus.lang_graph.routers.issue_answer_and_fix_need_test_router import (
    IssueAnswerAndFixNeedTestRouter,
)


def test_issue_answer_and_fix_need_test_router():
    router = IssueAnswerAndFixNeedTestRouter()
    state = {"run_test": True}
    assert router(state)

    state = {"run_test": False}
    assert not router(state)