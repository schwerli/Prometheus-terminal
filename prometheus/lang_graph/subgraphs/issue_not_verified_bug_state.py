from operator import add
from typing import Annotated, Mapping, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IssueNotVerifiedBugState(TypedDict):
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]
  number_of_candidate_patch: int
  max_refined_query_loop: int

  refined_query: str
  context_provider_messages: Annotated[Sequence[BaseMessage], add_messages]
  issue_bug_analyzer_messages: Annotated[Sequence[BaseMessage], add_messages]
  edit_messages: Annotated[Sequence[BaseMessage], add_messages]

  edit_patches: Annotated[Sequence[str], add]

  final_patch: str
