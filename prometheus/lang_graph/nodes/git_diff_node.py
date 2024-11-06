import logging

from prometheus.git.git_repository import GitRepository
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class GitDiffNode:
  def __init__(self):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.git_diff_node")

  def __call__(self, state: IssueAnswerAndFixState):
    git_repo = GitRepository(state["project_path"], None, copy_to_working_dir=False)
    patch = git_repo.get_diff()
    self._logger.debug(f"Generated patch:\n{patch}")
    return {"patch": patch}
