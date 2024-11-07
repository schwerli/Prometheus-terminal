from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from prometheus.lang_graph.subgraphs.issue_answer_state import IssueAnswerState


class IssueAnswerAndFixState(IssueAnswerState):
  project_path: str
  patch: str
  before_build_output: str
  before_test_output: str
  after_build_output: str
  after_test_output: str

  code_edit_messages: Annotated[Sequence[BaseMessage], add_messages]
  build_messages: Annotated[Sequence[BaseMessage], add_messages]
  build_summary: str
  build_success: bool
  test_messages: Annotated[Sequence[BaseMessage], add_messages]
