from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class BuildAndTestState(TypedDict):
  project_structure: str

  build_messages: Annotated[Sequence[BaseMessage], add_messages]
  exist_build: bool
  build_command_summary: str
  build_fail_log: str = ""

  test_messages: Annotated[Sequence[BaseMessage], add_messages]
  exist_test: bool
  test_command_summary: str
  test_fail_log: str = ""
