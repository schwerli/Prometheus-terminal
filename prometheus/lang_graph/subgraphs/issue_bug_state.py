from typing import Annotated, Mapping, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IssueBugState(TypedDict):
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]
  run_build: bool
  run_existing_test: bool

  bug_context: str

  reproduced_bug: bool
  reproduced_bug_file: str
  reproduced_bug_commands: Sequence[str]

  bug_fixing_messages: Annotated[Sequence[BaseMessage], add_messages]

  patch: str

  fixed_bug: bool
  reproducing_test_fail_log: str

  exist_build: bool
  build_command_summary: str
  build_fail_log: str = ""

  exist_test: bool
  test_command_summary: str
  existing_test_fail_log: str = ""
