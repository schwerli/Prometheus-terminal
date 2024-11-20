from typing import Mapping, Sequence, TypedDict


class IssueBugState(TypedDict):
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]

  bug_context: str

  reproduced_bug: bool
  reproduced_bug_file: str
  reproduced_bug_commands: Sequence[str]
