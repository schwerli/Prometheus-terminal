from enum import StrEnum
from typing import Mapping, Sequence, TypedDict


class IssueType(StrEnum):
  AUTO = "auto"
  BUG = "bug"
  FEATURE = "feature"
  DOCUMENTATION = "documentation"
  QUESTION = "question"


class IssueState(TypedDict):
  # Attributes provided by the user
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]
  issue_type: IssueType
  run_build: bool
  run_existing_test: bool

  reproduced_bug_file: str

  issue_response: str
