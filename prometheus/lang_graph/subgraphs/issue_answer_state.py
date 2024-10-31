from typing import Mapping, Sequence

from langchain_core.messages import BaseMessage

from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class IssueAnswerState(ContextProviderState):
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]
  issue_response: BaseMessage
