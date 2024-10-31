import functools
from typing import Optional

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.context_provider_node import ContextProviderNode
from prometheus.lang_graph.nodes.context_summary_node import ContextSummaryNode
from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class ContextProviderSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    context_provider_node = ContextProviderNode(model, kg, neo4j_driver)
    tool_node = ToolNode(tools=context_provider_node.tools, messages_key="context_messages")
    context_summary_node = ContextSummaryNode(model)

    workflow = StateGraph(ContextProviderState)
    workflow.add_node("context_provider_node", context_provider_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("context_summary_node", context_summary_node)
    workflow.add_conditional_edges(
      "context_provider_node",
      functools.partial(tools_condition, messages_key="context_messages"),
      {"tools": "tools", END: "context_summary_node"},
    )
    workflow.add_edge("tools", "context_provider_node")
    workflow.add_edge("context_summary_node", END)
    workflow.set_entry_point("context_provider_node")
    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(self, query: str, thread_id: Optional[str] = None):
    config = None
    if thread_id:
      config = {"configurable": {"thread_id": thread_id}}
    output_state = self.subgraph.invoke({"query": query}, config)
    return output_state["summary"]
