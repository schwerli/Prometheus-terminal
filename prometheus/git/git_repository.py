import os
import shutil
from pathlib import Path
from typing import Optional

from git import InvalidGitRepositoryError, Repo


class GitRepository:
  def __init__(
    self,
    address: str,
    working_directory: Path,
    copy_to_working_dir: bool = True,
    github_access_token: Optional[str] = None,
  ):
    if address.startswith("https://"):
      if github_access_token is None:
        raise ValueError("github_access_token is required for https repository")
      self.repo = self._clone_repository(address, github_access_token, working_directory)
      self._configure_remote_with_token()
    else:
      local_path = address
      if copy_to_working_dir:
        local_path = working_directory / os.path.basename(address)
        shutil.copytree(address, local_path)
      try:
        self.repo = Repo(local_path)
      except InvalidGitRepositoryError:
        self.repo = Repo.init(local_path)

  def _clone_repository(
    self, https_url: str, github_access_token: str, target_directory: Path
  ) -> Repo:
    self._original_https_url = https_url
    https_url = https_url.replace("https://", f"https://{github_access_token}@")
    repo_name = https_url.split("/")[-1].split(".")[0]
    local_path = target_directory / repo_name
    if local_path.exists():
      shutil.rmtree(local_path)

    return Repo.clone_from(https_url, local_path)
  
  def _configure_remote_with_token(self):
    """Configure the remote URL with the token for push operations."""
    if self.github_access_token and hasattr(self, '_original_https_url'):
        token_url = self._original_https_url.replace(
            "https://",
            f"https://{self.github_access_token}@"
        )
        self.repo.remote().set_url(token_url)

  def checkout_commit(self, commit_sha: str):
    self.repo.git.checkout(commit_sha)

  def switch_branch(self, branch_name: str):
    self.repo.git.checkout(branch_name)

  def pull(self):
    self.repo.git.pull()

  def get_diff(self) -> str:
    return self.repo.git.diff()

  def get_working_directory(self) -> str:
    return self.repo.working_dir

  def remove_repository(self):
    if self.repo is not None:
      shutil.rmtree(self.repo.working_dir)
      self.repo = None

  def create_and_push_branch(self, branch_name: str, commit_message: str):
    new_branch = self.repo.create_head(branch_name)
    new_branch.checkout()
    self.repo.git.add(A=True)
    self.repo.index.commit(commit_message)
    self.repo.git.push("--set-upstream", "origin", branch_name)
