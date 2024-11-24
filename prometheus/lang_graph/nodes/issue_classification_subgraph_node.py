from typing import Optional

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueState
from prometheus.lang_graph.subgraphs.issue_classification_subgraph import (
  IssueClassificationSubgraph,
)


class IssueClassificationSubgraphNode:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.issue_classification_subgraph = IssueClassificationSubgraph(
      model, kg, neo4j_driver, max_token_per_neo4j_result, thread_id, checkpointer
    )

  def __call__(self, state: IssueState):
    issue_type = self.issue_classification_subgraph.invoke(
      state["issue_title"], state["issue_body"], state["issue_comments"]
    )
    return {"issue_type": issue_type}
