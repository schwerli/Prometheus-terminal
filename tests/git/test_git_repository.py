import sys
from pathlib import Path
from unittest import mock

import pytest

from prometheus.git.git_repository import GitRepository
from tests.test_utils import test_project_paths
from tests.test_utils.fixtures import git_repo_fixture  # noqa: F401


@pytest.mark.skipif(
  sys.platform.startswith("win"),
  reason="Test fails on Windows because of cptree in git_repo_fixture",
)
def test_init_with_https_url(git_repo_fixture):  # noqa: F811
  with mock.patch("git.Repo.clone_from") as mock_clone_from, mock.patch("shutil.rmtree"):
    repo = git_repo_fixture
    mock_clone_from.return_value = repo

    access_token = "access_token"
    https_url = "https://github.com/foo/bar.git"
    target_directory = test_project_paths.TEST_PROJECT_PATH

    GitRepository(
      address=https_url, working_directory=target_directory, github_access_token=access_token
    )

    mock_clone_from.assert_called_once_with(
      f"https://{access_token}@github.com/foo/bar.git", target_directory / "bar"
    )


@pytest.mark.skipif(
  sys.platform.startswith("win"),
  reason="Test fails on Windows because of cptree in git_repo_fixture",
)
def test_checkout_commit(git_repo_fixture):  # noqa: F811
  local_path = str(test_project_paths.TEST_PROJECT_PATH)

  git_repo = GitRepository(
    address=local_path, working_directory=Path("/foo/bar"), copy_to_working_dir=False
  )

  commit_sha = "293551b7bd9572b63018c9ed2bccea0f37726805"
  assert git_repo.repo.head.commit.hexsha != commit_sha
  git_repo.checkout_commit(commit_sha)
  assert git_repo.repo.head.commit.hexsha == commit_sha


@pytest.mark.skipif(
  sys.platform.startswith("win"),
  reason="Test fails on Windows because of cptree in git_repo_fixture",
)
def test_switch_branch(git_repo_fixture):  # noqa: F811
  local_path = str(test_project_paths.TEST_PROJECT_PATH)

  git_repo = GitRepository(
    address=local_path, working_directory=Path("/foo/bar"), copy_to_working_dir=False
  )

  branch_name = "dev"
  assert git_repo.repo.active_branch.name != branch_name
  git_repo.switch_branch(branch_name)
  assert git_repo.repo.active_branch.name == branch_name


@pytest.mark.skipif(
  sys.platform.startswith("win"),
  reason="Test fails on Windows because of cptree in git_repo_fixture",
)
def test_remove_repository(git_repo_fixture):  # noqa: F811
  with mock.patch("shutil.rmtree") as mock_rmtree:
    local_path = str(test_project_paths.TEST_PROJECT_PATH)

    git_repo = GitRepository(
      address=local_path, working_directory=Path("/foo/bar"), copy_to_working_dir=False
    )

    git_repo.remove_repository()

    mock_rmtree.assert_called_once_with(git_repo_fixture.working_dir)
    assert git_repo.repo is None
