from sqlmodel import Field, SQLModel


class Repository(SQLModel, table=True):
    id: int = Field(primary_key=True, description="ID")
    repo: str = Field(
        index=True, max_length=100, description="The repository owner/name identifier from GitHub."
    )
    commit_id: str = Field(
        index=True,
        nullable=True,
        min_length=40,
        max_length=40,
        description="The commit id of the repository.",
    )
    playground_path: str = Field(
        unique=True,
        max_length=100,
        description="The playground path of the repository where the repository was cloned.",
    )
    user_id: int = Field(
        index=True, nullable=True, description="The ID of the user who upload this repository."
    )
    is_cleaned: bool = Field(default=False, description="Whether the repository is cleaned or not.")
    kg_root_node_id: int = Field(
        index=True, unique=True, description="The ID of the root node of the knowledge graph."
    )
