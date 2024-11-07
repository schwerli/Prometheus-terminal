from typing import Annotated, Literal, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from prometheus.lang_graph.subgraphs.issue_answer_state import IssueAnswerState


class IssueAnswerAndFixState(IssueAnswerState):
  project_path: str
  run_build: bool
  run_test: bool

  build_messages: Annotated[Sequence[BaseMessage], add_messages]
  exist_build: bool
  build_summary: str
  build_success: bool

  test_messages: Annotated[Sequence[BaseMessage], add_messages]
  exist_test: bool
  test_summary: str
  test_success: bool

  last_edit_failure_cause: Literal["Build", "Test"]

  code_edit_messages: Annotated[Sequence[BaseMessage], add_messages]
  patch: str
