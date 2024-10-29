from typing import Optional

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.context_provider_node import (
  ContextProviderNode,
  ContextProviderState,
)


class ContextProviderSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    agent_node = ContextProviderNode(model, kg, neo4j_driver)
    tools = agent_node.tools
    tool_node = ToolNode(tools=tools)

    workflow = StateGraph(ContextProviderState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_conditional_edges(
      "agent",
      tools_condition,
    )
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")
    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(self, query: str, thread_id: Optional[str] = None):
    config = None
    if thread_id:
      config = {"configurable": {"thread_id": thread_id}}
    output_state = self.subgraph.invoke({"query": query}, config)
    return output_state["messages"][-1].content
