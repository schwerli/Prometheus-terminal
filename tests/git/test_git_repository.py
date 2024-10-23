from unittest import mock
from pathlib import Path

import git

from prometheus.git.git_repository import GitRepository
from tests.test_utils import test_project_paths
from tests.test_utils.fixtures import git_repo_fixture


def test_clone_repository(git_repo_fixture):
  with mock.patch("git.Repo.clone_from") as mock_clone_from:
    repo = git_repo_fixture
    mock_clone_from.return_value = repo

    access_token = "access_token"
    https_url = "https://github.com/foo/bar.git"
    target_directory = test_project_paths.TEST_PROJECT_PATH
    git_repo = GitRepository(github_access_token=access_token)
    git_repo.clone_repository(https_url, target_directory)

    mock_clone_from.assert_called_once_with(
      f"https://{access_token}@github.com/foo/bar.git",
      test_project_paths.TEST_PROJECT_PATH / "bar"
    )

def test_checkout_commit(git_repo_fixture):
  with mock.patch("git.Repo.clone_from") as mock_clone_from:
    repo = git_repo_fixture
    mock_clone_from.return_value = repo

    access_token = "access_token"
    https_url = "https://github.com/foo/bar.git"
    target_directory = test_project_paths.TEST_PROJECT_PATH
    git_repo = GitRepository(github_access_token=access_token)
    git_repo.clone_repository(https_url, target_directory)


    commit_sha = "293551b7bd9572b63018c9ed2bccea0f37726805"
    assert repo.head.commit.hexsha != commit_sha
    git_repo.checkout_commit(commit_sha)
    assert repo.head.commit.hexsha == commit_sha

    repo.git.checkout("master")

def test_switch_branch(git_repo_fixture):
  original_execute = git.Git.execute
  with mock.patch("git.Repo.clone_from") as mock_clone_from, \
    mock.patch("git.Git.execute") as mock_execute:
    repo = git_repo_fixture
    mock_clone_from.return_value = repo

    access_token = "access_token"
    https_url = "https://github.com/foo/bar.git"
    target_directory = test_project_paths.TEST_PROJECT_PATH
    git_repo = GitRepository(github_access_token=access_token)
    git_repo.clone_repository(https_url, target_directory)

    def execute_side_effect(command, *args, **kwargs):
      if command == ['git', 'pull']:
        return "Done"
      else:
        return original_execute(repo.git, command, *args, **kwargs)

    mock_execute.side_effect = execute_side_effect

    branch_name = "dev"
    assert repo.active_branch.name != branch_name
    git_repo.switch_branch(branch_name)
    assert repo.active_branch.name == branch_name

    repo.git.checkout("master")
