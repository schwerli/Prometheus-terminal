import logging

from langchain_core.messages import HumanMessage

from prometheus.lang_graph.subgraphs.context_retrieval_state import ContextRetrievalState


class ContextQueryMessageNode:
  def __init__(self):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.context_query_message_node")

  def __call__(self, state: ContextRetrievalState):
    human_message = HumanMessage(state["query"])
    self._logger.debug(f"Sending query to ContextProviderNode:\n{human_message}")
    return {"context_provider_messages": [human_message]}
