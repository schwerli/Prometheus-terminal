from prometheus.git.git_repository import GitRepository
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class GitDiffNode:
  def __init__(self, git_repo: GitRepository):
    self.git_repo = git_repo

  def __call__(self, state: IssueAnswerAndFixState):
    return {"patch": self.git_repo.get_diff()}
