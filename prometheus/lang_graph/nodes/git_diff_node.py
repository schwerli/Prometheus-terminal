from prometheus.git.git_repository import GitRepository
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class GitDiffNode:
  def __call__(self, state: IssueAnswerAndFixState):
    git_repo = GitRepository(state["project_path"], None, copy_to_working_dir=False)
    return {"patch": git_repo.get_diff()}
