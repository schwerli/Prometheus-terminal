from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueType
from prometheus.lang_graph.nodes.issue_question_responder_node import IssueQuestionResponderNode
from prometheus.lang_graph.nodes.issue_to_context_node import IssueToContextNode
from prometheus.lang_graph.subgraphs.issue_question_state import IssueQuestionState


class IssueQuestionSubgrapgh:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.thread_id = thread_id
    issue_to_question_context_node = IssueToContextNode(
      IssueType.QUESTION,
      model,
      kg,
      neo4j_driver,
      max_token_per_neo4j_result,
      thread_id,
      checkpointer,
    )
    issue_question_responder_node = IssueQuestionResponderNode(model)

    workflow = StateGraph(IssueQuestionState)

    workflow.add_node("issue_to_question_context_node", issue_to_question_context_node)
    workflow.add_node("issue_question_responder_node", issue_question_responder_node)

    workflow.set_entry_point("issue_to_question_context_node")
    workflow.add_edge("issue_to_question_context_node", "issue_question_responder_node")
    workflow.add_edge("issue_question_responder_node", END)

    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(self, issue_title: str, issue_body: str, issue_comments: Sequence[Mapping[str, str]]):
    config = None
    if self.thread_id:
      config = {"configurable": {"thread_id": self.thread_id}}

    output_state = self.subgraph.invoke(
      {"issue_title": issue_title, "issue_body": issue_body, "issue_comments": issue_comments},
      config,
    )
    return output_state["issue_response"]
