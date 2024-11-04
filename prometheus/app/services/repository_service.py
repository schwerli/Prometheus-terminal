from pathlib import Path
from typing import Optional

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph


class RepositoryService:
  def __init__(
    self,
    kg_service: KnowledgeGraphService,
    github_token: str,
    max_ast_depth: int,
    working_dir: Path,
  ):
    self.kg_service = kg_service
    self.git_repo = GitRepository(github_token)
    self.max_ast_depth = max_ast_depth
    self.working_dir = working_dir

  def upload_local_repository(self, path: Path):
    self.kg_service.clear()
    kg = KnowledgeGraph(self.max_ast_depth)
    kg.build_graph(path)
    self.kg_service.kg = kg
    self.kg_service.kg_handler.write_knowledge_graph(kg)

  def upload_github_repo(self, https_url: str, commit_id: Optional[str] = None):
    if self._should_skip_upload(https_url, commit_id):
      return

    self.kg_service.clear()
    if self.git_repo.has_repository():
      self.git_repo.remove_repository()

    target_directory = self.working_dir / "repositories"
    target_directory.mkdir(parents=True, exist_ok=True)

    saved_path = self.git_repo.clone_repository(https_url, target_directory)
    if commit_id:
      self.git_repo.checkout_commit(commit_id)

    kg = KnowledgeGraph(self.max_ast_depth)
    kg.build_graph(saved_path)
    self.kg_service.kg = kg
    self.kg_service.kg_handler.write_knowledge_graph(kg)

  def _should_skip_upload(self, https_url: str, commit_id: Optional[str]) -> bool:
    kg = self.kg_service.kg
    return (
      kg is not None
      and kg.is_built_from_github()
      and commit_id
      and kg.get_codebase_https_url() == https_url
      and kg.get_codebase_commit_id() == commit_id
    )
