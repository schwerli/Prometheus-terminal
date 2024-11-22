import functools
from typing import Optional

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.context_provider_node import ContextProviderNode
from prometheus.lang_graph.nodes.context_refine_node import ContextRefineNode
from prometheus.lang_graph.nodes.context_summary_node import ContextSummaryNode
from prometheus.lang_graph.nodes.reset_messages_node import ResetMessagesNode
from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class ContextProviderSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    context_provider_node = ContextProviderNode(model, kg, neo4j_driver, max_token_per_neo4j_result)
    context_provider_tools = ToolNode(
      tools=context_provider_node.tools,
      name="context_provider_tools",
      messages_key="context_provider_messages",
    )
    context_refine_node = ContextRefineNode(model)
    reset_context_provider_messages_node = ResetMessagesNode("context_provider_messages")
    context_summary_node = ContextSummaryNode(model)

    workflow = StateGraph(ContextProviderState)
    workflow.add_node("context_provider_node", context_provider_node)
    workflow.add_node("context_provider_tools", context_provider_tools)
    workflow.add_node("context_refine_node", context_refine_node)
    workflow.add_node("reset_context_provider_messages_node", reset_context_provider_messages_node)
    workflow.add_node("context_summary_node", context_summary_node)

    workflow.set_entry_point("context_provider_node")

    workflow.add_conditional_edges(
      "context_provider_node",
      functools.partial(tools_condition, messages_key="context_provider_messages"),
      {"tools": "context_provider_tools", END: "context_refine_node"},
    )
    workflow.add_edge("context_provider_tools", "context_provider_node")
    workflow.add_conditional_edges(
      "context_refine_node",
      lambda state: state["has_sufficient_context"] and state["remaining_steps"] > 10,
      {True: "context_summary_node", False: "reset_context_provider_messages_node"},
    )
    workflow.add_edge("reset_context_provider_messages_node", "context_provider_node")
    workflow.add_edge("context_summary_node", END)

    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(self, query: str, thread_id: Optional[str] = None):
    config = None
    if thread_id:
      config = {"configurable": {"thread_id": thread_id}}
    output_state = self.subgraph.invoke({"original_query": query}, config)
    return output_state["summary"]
