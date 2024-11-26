from typing import Annotated, Mapping, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IssueClassificationState(TypedDict):
  # Attributes provided by the user
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]

  # Attributes generated and by the subgraph
  context_provider_messages: Annotated[Sequence[BaseMessage], add_messages]
  issue_type: str
