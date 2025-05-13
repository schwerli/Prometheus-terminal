"""Service for managing repository (GitHub or local) operations."""

import shutil
import uuid
from pathlib import Path
from typing import Optional

from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.git.git_repository import GitRepository


class RepositoryService:
    """Manages repository operations.

    This service provides functionality for Git repository operations including
    cloning repositories, managing commits, pushing changes, and maintaining
    a clean working directory. It integrates with a knowledge graph service
    to track repository state and avoid redundant operations.
    """

    def __init__(
        self,
        kg_service: KnowledgeGraphService,
        working_dir: str,
    ):
        """Initializes the repository service.

        Args:
          kg_service: Knowledge graph service instance for codebase tracking.
          working_dir: Base directory for repository operations. A 'repositories'
              subdirectory will be created under this path.
        """
        self.kg_service = kg_service
        self.target_directory = Path(working_dir) / "repositories"
        self.target_directory.mkdir(parents=True, exist_ok=True)
        self.git_repo = self._load_existing_git_repo()

    def _load_existing_git_repo(self):
        if self.kg_service.get_local_path() and self.kg_service.get_local_path().exists():
            return GitRepository(
                str(self.kg_service.get_local_path()), None, copy_to_working_dir=False
            )
        return None

    def get_working_dir(self):
        if self.git_repo:
            return self.git_repo.get_working_directory()
        return None

    def clone_github_repo(
        self, github_token: str, https_url: str, commit_id: Optional[str] = None
    ) -> Path:
        """Clones a GitHub repository to the local workspace.

        Clones the specified repository and optionally checks out a specific commit.
        If the repository is already present and matches the requested state,
        the operation may be skipped.

        Args:
            github_token: GitHub access token for authentication.
            https_url: HTTPS URL of the GitHub repository.
            commit_id: Optional specific commit to check out.

        Returns:
            Path to the local repository directory.
        """
        self.git_repo = GitRepository(
            https_url, self.target_directory, github_access_token=github_token
        )
        if commit_id:
            self.git_repo.checkout_commit(commit_id)
        local_path = self.git_repo.get_working_directory()
        return local_path

    def push_change_to_remote(self, commit_message: str, patch: str):
        """Pushes local changes to a new remote branch.

        Creates a new branch with a unique name, commits the current changes,
        and pushes them to the remote repository. Branch names are prefixed with
        'prometheus_fix_' and include a unique identifier.

        Args:
          commit_message: Message to use for the commit.

        Returns:
          Name of the created branch.
        """
        branch_name = f"prometheus_fix_{uuid.uuid4().hex[:10]}"
        self.git_repo.create_and_push_branch(branch_name, commit_message, patch)
        return branch_name

    def clean(self):
        self.git_repo = None
        shutil.rmtree(self.target_directory)
        self.target_directory.mkdir(parents=True)
