from typing import Annotated, Mapping, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class BugReproductionState(TypedDict):
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]

  context_provider_messages: Annotated[Sequence[BaseMessage], add_messages]

  bug_reproducing_write_messages: Annotated[Sequence[BaseMessage], add_messages]
  bug_reproducing_file_messages: Annotated[Sequence[BaseMessage], add_messages]
  bug_reproducing_execute_messages: Annotated[Sequence[BaseMessage], add_messages]

  reproduced_bug: bool
  reproduced_bug_failure_log: str
  reproduced_bug_file: str
  reproduced_bug_commands: Sequence[str]
