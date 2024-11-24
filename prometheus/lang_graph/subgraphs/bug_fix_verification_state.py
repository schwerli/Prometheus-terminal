from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class BugFixVerficationState(TypedDict):
  reproduced_bug_file: str
  reproduced_bug_commands: Sequence[str]

  bug_fix_verify_messages: Annotated[Sequence[BaseMessage], add_messages]

  reproducing_test_passed: bool
  reproducing_test_fail_log: str
