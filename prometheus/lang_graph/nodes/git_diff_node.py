"""Git diff generation for project changes.

This module provides functionality to generate Git diffs for changes made to a project,
typically used to track modifications made during automated issue resolution or code
fixes. It uses the GitRepository class to access Git operations and generate patch
output.
"""

import logging
from typing import Dict

from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph


class GitDiffNode:
  """Generates Git diffs for project modifications.

  This class handles the generation of Git diffs to track changes made to a project.
  It works with a GitRepository instance to access the project's Git operations
  and create patch output. The node is typically used as part of an automated
  workflow to capture code modifications made during issue resolution.
  """

  def __init__(self, kg: KnowledgeGraph, exclude_reproduced_bug_file: bool = False):
    self.kg = kg
    self.exclude_reproduced_bug_file = exclude_reproduced_bug_file
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.git_diff_node")

  def __call__(self, state: Dict):
    """Generates a Git diff for the current project state.

    Creates a Git repository instance for the project path specified in the
    state and generates a diff of any uncommitted changes.

    Args:
      state: Current state containing project information, including the
        project_path key specifying the Git repository location.

    Returns:
      Dictionary that update the state containing:
      - patch: String containing the Git diff output showing all changes made to the project.
    """
    git_repo = GitRepository(self.kg.get_local_path(), None, copy_to_working_dir=False)
    if self.exclude_reproduced_bug_file:
      patch = git_repo.get_diff([state["reproduced_bug_file"]])
    else:
      patch = git_repo.get_diff()
    self._logger.debug(f"Generated patch:\n{patch}")
    return {"patch": patch}
