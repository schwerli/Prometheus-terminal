from typing import Annotated, Mapping, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IssueBugState(TypedDict):
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]

  bug_context: str

  reproduced_bug: bool
  reproduced_bug_file: str
  reproduced_bug_commands: Sequence[str]

  bug_fixing_messages: Annotated[Sequence[BaseMessage], add_messages]
