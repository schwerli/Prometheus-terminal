from git import Repo

from pathlib import Path


class GitRepository:
  def __init__(self, github_access_token: str):
    self._github_access_token = github_access_token
    self._original_https_url = None
    self._repo = None

  def clone_repository(self, https_url: str, target_directory: Path):
    self._original_https_url = https_url
    https_url = https_url.replace("https://", f"https://{self._github_access_token}@")
    repo_name = https_url.split("/")[-1].split(".")[0]
    local_path = target_directory / repo_name
    self._repo = Repo.clone_from(https_url, local_path)

  def checkout_commit(self, commit_sha: str):
    self._repo.git.checkout(commit_sha)

  def switch_branch(self, branch_name: str):
    self._repo.git.checkout(branch_name)
    self._repo.git.pull()

  def pull(self):
    self._repo.git.pull()
