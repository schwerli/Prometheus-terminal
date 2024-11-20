from pathlib import Path
from typing import Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueState, IssueType
from prometheus.lang_graph.nodes.issue_classification_subgraph_node import (
  IssueClassificationSubgraphNode,
)
from prometheus.lang_graph.nodes.issue_question_subgraph_node import IssueQuestionSubgraphNode
from prometheus.lang_graph.nodes.noop_node import NoopNode


class IssueGraph:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    local_path: Path,
    dockerfile_content: Optional[str] = None,
    image_name: Optional[str] = None,
    workdir: Optional[str] = None,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    issue_type_branch_node = NoopNode()
    issue_classification_subgraph_node = IssueClassificationSubgraphNode(
      model, kg, neo4j_driver, max_token_per_neo4j_result, thread_id, checkpointer
    )
    issue_question_subgraph_node = IssueQuestionSubgraphNode(
      model, kg, neo4j_driver, max_token_per_neo4j_result, thread_id, checkpointer
    )

    workflow = StateGraph(IssueState)

    workflow.add_node("issue_type_branch_node", issue_type_branch_node)
    workflow.add_node("issue_classification_subgraph_node", issue_classification_subgraph_node)
    workflow.add_node("issue_question_subgraph_node", issue_question_subgraph_node)

    workflow.set_entry_point("issue_type_branch_node")
    workflow.add_conditional_edges(
      "issue_type_branch_node",
      lambda state: state["issue_type"],
      {
        IssueType.AUTO: "issue_classification_subgraph_node",
        IssueType.QUESTION: "issue_question_subgraph_node",
      },
    )

    self.graph = workflow.compile(checkpointer=checkpointer)
