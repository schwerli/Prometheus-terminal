from enum import StrEnum
from typing import Annotated, Mapping, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class ResponseModeEnum(StrEnum):
  AUTO = "auto"
  ONLY_ANSWER = "only_answer"
  ANSWER_AND_FIX = "answer_and_fix"


class IssueAnswerAndFixState(ContextProviderState):
  # Attributes provided by the user
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]
  response_mode: ResponseModeEnum
  run_build: bool
  run_test: bool

  # All attributes generated and used by the subgraph
  project_path: str
  project_structure: str

  require_edit: bool

  build_messages: Annotated[Sequence[BaseMessage], add_messages]
  exist_build: bool
  build_command_summary: str
  build_fail_log: str = ""

  test_messages: Annotated[Sequence[BaseMessage], add_messages]
  exist_test: bool
  test_command_summary: str
  test_fail_log: str = ""

  code_edit_messages: Annotated[Sequence[BaseMessage], add_messages]
  patch: str

  edit_reviewer_messages: Annotated[Sequence[BaseMessage], add_messages]
  reviewer_approved: bool
  reviewer_comments: str

  issue_response: str
