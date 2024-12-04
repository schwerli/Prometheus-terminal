from typing import Mapping, Sequence, TypedDict


class IssueBugState(TypedDict):
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]
  run_build: bool
  run_existing_test: bool

  reproduced_bug: bool
  reproduced_bug_file: str
  reproduced_bug_commands: Sequence[str]

  edit_patch: str

  reproducing_test_fail_log: str

  exist_build: bool
  build_fail_log: str

  exist_test: bool
  existing_test_fail_log: str

  issue_response: str
