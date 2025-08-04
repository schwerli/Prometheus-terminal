from pydantic import BaseModel


class IssueResponse(BaseModel):
    patch: str
    passed_reproducing_test: bool
    passed_build: bool
    passed_existing_test: bool
    issue_response: str
    remote_branch_name: str | None = None
