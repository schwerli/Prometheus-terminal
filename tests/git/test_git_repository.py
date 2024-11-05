from pathlib import Path
from unittest import mock

import git
import pytest

from prometheus.git.git_repository import GitRepository
from tests.test_utils import test_project_paths
from tests.test_utils.fixtures import git_repo_fixture  # noqa: F401


@pytest.mark.skip
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


@pytest.mark.skip
def test_init_with_local_path(git_repo_fixture):  # noqa: F811
  with (
    mock.patch("shutil.copytree") as mock_copytree,
    mock.patch("git.Repo") as mock_repo,
    mock.patch("os.path.exists", return_value=True),
  ):
    mock_repo.return_value = git_repo_fixture
    local_path = "/path/to/local/repo"
    target_directory = test_project_paths.TEST_PROJECT_PATH

    GitRepository(address=local_path, working_directory=target_directory, copy_to_working_dir=True)

    expected_target = target_directory / Path(local_path).name
    mock_copytree.assert_called_once_with(local_path, expected_target)
    mock_repo.assert_called_once_with(expected_target)


@pytest.mark.skip
def test_init_with_local_path_no_copy(git_repo_fixture):  # noqa: F811
  with mock.patch("git.Repo") as mock_repo, mock.patch("os.path.exists", return_value=True):
    mock_repo.return_value = git_repo_fixture
    local_path = "/path/to/local/repo"
    target_directory = test_project_paths.TEST_PROJECT_PATH

    GitRepository(address=local_path, working_directory=target_directory, copy_to_working_dir=False)

    mock_repo.assert_called_once_with(local_path)


@pytest.mark.skip
def test_init_with_https_url_no_token():
  https_url = "https://github.com/foo/bar.git"
  target_directory = test_project_paths.TEST_PROJECT_PATH

  with pytest.raises(ValueError, match="github_access_token is required for https repository"):
    GitRepository(address=https_url, working_directory=target_directory, github_access_token=None)


@pytest.mark.skip
def test_checkout_commit(git_repo_fixture):  # noqa: F811
  with mock.patch("git.Repo.clone_from") as mock_clone_from, mock.patch("shutil.rmtree"):
    repo = git_repo_fixture
    mock_clone_from.return_value = repo

    git_repo = GitRepository(
      address="https://github.com/foo/bar.git",
      working_directory=test_project_paths.TEST_PROJECT_PATH,
      github_access_token="access_token",
    )

    commit_sha = "293551b7bd9572b63018c9ed2bccea0f37726805"
    assert repo.head.commit.hexsha != commit_sha
    git_repo.checkout_commit(commit_sha)
    assert repo.head.commit.hexsha == commit_sha


@pytest.mark.skip
def test_switch_branch(git_repo_fixture):  # noqa: F811
  original_execute = git.Git.execute
  with (
    mock.patch("git.Repo.clone_from") as mock_clone_from,
    mock.patch("shutil.rmtree"),
    mock.patch("git.Git.execute") as mock_execute,
  ):
    repo = git_repo_fixture
    mock_clone_from.return_value = repo

    git_repo = GitRepository(
      address="https://github.com/foo/bar.git",
      working_directory=test_project_paths.TEST_PROJECT_PATH,
      github_access_token="access_token",
    )

    def execute_side_effect(command, *args, **kwargs):
      if command == ["git", "pull"]:
        return "Done"
      else:
        return original_execute(repo.git, command, *args, **kwargs)

    mock_execute.side_effect = execute_side_effect

    branch_name = "dev"
    assert repo.active_branch.name != branch_name
    git_repo.switch_branch(branch_name)
    assert repo.active_branch.name == branch_name


@pytest.mark.skip
def test_remove_repository(git_repo_fixture):  # noqa: F811
  with (
    mock.patch("shutil.rmtree") as mock_rmtree,
    mock.patch("git.Repo") as mock_repo,
    mock.patch("os.path.exists", return_value=True),
  ):
    mock_repo.return_value = git_repo_fixture

    git_repo = GitRepository(
      address="/path/to/local/repo",
      working_directory=test_project_paths.TEST_PROJECT_PATH,
      copy_to_working_dir=False,
    )

    git_repo.remove_repository()

    mock_rmtree.assert_called_once_with(git_repo_fixture.working_dir)
    assert git_repo.repo is None
