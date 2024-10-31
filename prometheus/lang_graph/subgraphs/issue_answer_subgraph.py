from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.issue_responder_node import IssueResponderNode
from prometheus.lang_graph.nodes.issue_to_query_node import IssueToQueryNode
from prometheus.lang_graph.subgraphs.context_provider_subgraph import ContextProviderSubgraph
from prometheus.lang_graph.subgraphs.issue_answer_state import IssueAnswerState


class IssueAnswerSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    issue_to_query_node = IssueToQueryNode()
    context_provider_subgraph = ContextProviderSubgraph(
      model, kg, neo4j_driver, checkpointer
    ).subgraph
    issue_responder_node = IssueResponderNode(model)

    workflow = StateGraph(IssueAnswerState)
    workflow.add_node("issue_to_query_node", issue_to_query_node)
    workflow.add_node("context_provider_subgraph", context_provider_subgraph)
    workflow.add_node("issue_responder_node", issue_responder_node)
    workflow.add_edge("issue_to_query_node", "context_provider_subgraph")
    workflow.add_edge("context_provider_subgraph", "issue_responder_node")
    workflow.add_edge("issue_responder_node", END)
    workflow.set_entry_point("issue_to_query_node")
    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(
    self,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    thread_id: Optional[str] = None,
  ):
    config = None
    if thread_id:
      config = {"configurable": {"thread_id": thread_id}}
    output_state = self.subgraph.invoke(
      {"issue_title": issue_title, "issue_body": issue_body, "issue_comments": issue_comments},
      config,
    )
    return output_state["issue_response"]
