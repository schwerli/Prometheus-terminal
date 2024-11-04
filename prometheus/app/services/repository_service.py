import shutil
from pathlib import Path
from typing import Optional

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.git.git_repository import GitRepository


class RepositoryService:
  def __init__(
    self,
    kg_service: KnowledgeGraphService,
    github_token: str,
    working_dir: str,
  ):
    self.kg_service = kg_service
    self.git_repo = GitRepository(github_token)
    self.target_directory = Path(working_dir) / "repositories"
    self.target_directory.mkdir(parents=True, exist_ok=True)

  def clone_github_repo(self, https_url: str, commit_id: Optional[str] = None):
    if self._should_skip_upload(https_url, commit_id):
      return

    if self.git_repo.has_repository():
      self.git_repo.remove_repository()

    saved_path = self.git_repo.clone_repository(https_url, self.target_directory)
    if commit_id:
      self.git_repo.checkout_commit(commit_id)
    return saved_path

  def _should_skip_upload(self, https_url: str, commit_id: Optional[str]) -> bool:
    kg = self.kg_service.kg
    return (
      kg is not None
      and kg.is_built_from_github()
      and commit_id
      and kg.get_codebase_https_url() == https_url
      and kg.get_codebase_commit_id() == commit_id
    )

  def clean_working_directory(self):
    shutil.rmtree(self.target_directory)
    self.target_directory.mkdir(parents=True)
