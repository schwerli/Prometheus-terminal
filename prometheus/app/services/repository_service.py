import shutil
import uuid
from pathlib import Path
from typing import Optional

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.git.git_repository import GitRepository


class RepositoryService:
  def __init__(
    self,
    kg_service: KnowledgeGraphService,
    working_dir: str,
  ):
    self.kg_service = kg_service
    self.target_directory = Path(working_dir) / "repositories"
    self.target_directory.mkdir(parents=True, exist_ok=True)
    self.local_path = None

  def clone_github_repo(
    self, github_token: str, https_url: str, commit_id: Optional[str] = None
  ) -> Path:
    if self._should_skip_upload(https_url, commit_id):
      return self.local_path

    git_repo = GitRepository(https_url, self.target_directory, github_access_token=github_token)
    if commit_id:
      git_repo.checkout_commit(commit_id)
    self.local_path = Path(git_repo.get_working_directory())
    return self.local_path

  def _should_skip_upload(self, https_url: str, commit_id: Optional[str]) -> bool:
    kg = self.kg_service.kg
    return (
      self.local_path is not None
      and kg is not None
      and kg.is_built_from_github()
      and commit_id
      and kg.get_codebase_https_url() == https_url
      and kg.get_codebase_commit_id() == commit_id
    )

  def push_change_to_remote(self, commit_message: str):
    git_repo = GitRepository(str(self.local_path.absolute()), None, False)
    branch_name = f"prometheus_fix_{uuid.uuid4().hex[:10]}"
    git_repo.create_and_push_branch(branch_name, commit_message)
    return branch_name

  def clean_working_directory(self):
    shutil.rmtree(self.target_directory)
    self.target_directory.mkdir(parents=True)
