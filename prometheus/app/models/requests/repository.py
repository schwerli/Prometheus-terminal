from pydantic import BaseModel, Field


class UploadRepositoryRequest(BaseModel):
    https_url: str = Field(description="The URL of the repository", max_length=100)
    commit_id: str | None = Field(
        default=None,
        description="The commit id of the repository, "
        "if not provided, the latest commit in the main branch will be used.",
        min_length=40,
        max_length=40,
    )
    github_token: str | None = Field(
        default=None,
        description="Optional GitHub token for repository clone",
        max_length=100,
    )
