from typing import Annotated, Mapping, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class IssueAnswerAndFixState(ContextProviderState):
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]
  only_answer: bool
  run_build: bool
  run_test: bool

  project_path: str
  project_structure: str

  build_messages: Annotated[Sequence[BaseMessage], add_messages]
  exist_build: bool
  build_command_summary: str
  build_fail_log: str

  test_messages: Annotated[Sequence[BaseMessage], add_messages]
  exist_test: bool
  test_command_summary: str
  test_fail_log: str

  code_edit_messages: Annotated[Sequence[BaseMessage], add_messages]
  patch: str

  issue_response: str
