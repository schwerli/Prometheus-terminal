from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps


class ContextProviderState(TypedDict):
  original_query: str

  remaining_steps: RemainingSteps

  context_provider_messages: Annotated[Sequence[BaseMessage], add_messages]
  all_context_provider_responses: Annotated[Sequence[BaseMessage], add_messages]

  has_sufficient_context: bool
  refined_query: str

  summary: str
