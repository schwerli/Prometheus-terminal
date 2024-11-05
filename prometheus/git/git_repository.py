import shutil
from pathlib import Path

from git import Repo


class GitRepository:
  def __init__(self, github_access_token: str):
    self._github_access_token = github_access_token
    self._original_https_url = None
    self._repo = None

  def clone_repository(self, https_url: str, target_directory: Path) -> Path:
    if self._original_https_url == https_url and self._repo is not None:
      self.pull()

    self._original_https_url = https_url
    https_url = https_url.replace("https://", f"https://{self._github_access_token}@")
    repo_name = https_url.split("/")[-1].split(".")[0]
    local_path = target_directory / repo_name
    if local_path.exists():
      shutil.rmtree(local_path)

    self._repo = Repo.clone_from(https_url, local_path)
    return local_path

  def checkout_commit(self, commit_sha: str):
    self._repo.git.checkout(commit_sha)

  def switch_branch(self, branch_name: str):
    self._repo.git.checkout(branch_name)
    self._repo.git.pull()

  def pull(self):
    self._repo.git.pull()

  def get_diff(self) -> str:
    return self._repo.git.diff()

  def has_repository(self) -> bool:
    return self._repo is not None

  def remove_repository(self):
    if self._repo is not None:
      shutil.rmtree(self._repo.working_dir)
      self._repo = None
      self._original_https_url = None
