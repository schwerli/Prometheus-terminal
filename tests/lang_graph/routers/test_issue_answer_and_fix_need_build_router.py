from prometheus.lang_graph.routers.issue_answer_and_fix_need_build_router import (
  IssueAnswerAndFixNeedBuildRouter,
)


def test_issue_answer_and_fix_need_build_router():
  router = IssueAnswerAndFixNeedBuildRouter()
  state = {"run_build": True}
  assert router(state)

  state = {"run_build": False}
  assert not router(state)
