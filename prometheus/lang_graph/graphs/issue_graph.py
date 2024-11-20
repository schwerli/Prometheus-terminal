from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueState, IssueType
from prometheus.lang_graph.nodes.issue_bug_subgraph_node import IssueBugSubgraphNode
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
    container: BaseContainer,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.thread_id = thread_id

    issue_type_branch_node = NoopNode()
    issue_classification_subgraph_node = IssueClassificationSubgraphNode(
      model, kg, neo4j_driver, max_token_per_neo4j_result, thread_id, checkpointer
    )
    issue_bug_subgraph_node = IssueBugSubgraphNode(
      model,
      container,
      kg,
      neo4j_driver,
      max_token_per_neo4j_result,
      build_commands,
      test_commands,
      thread_id,
      checkpointer,
    )
    issue_question_subgraph_node = IssueQuestionSubgraphNode(
      model, kg, neo4j_driver, max_token_per_neo4j_result, thread_id, checkpointer
    )

    workflow = StateGraph(IssueState)

    workflow.add_node("issue_type_branch_node", issue_type_branch_node)
    workflow.add_node("issue_classification_subgraph_node", issue_classification_subgraph_node)
    workflow.add_node("issue_bug_subgraph_node", issue_bug_subgraph_node)
    workflow.add_node("issue_question_subgraph_node", issue_question_subgraph_node)

    workflow.set_entry_point("issue_type_branch_node")
    workflow.add_conditional_edges(
      "issue_type_branch_node",
      lambda state: state["issue_type"],
      {
        IssueType.AUTO: "issue_classification_subgraph_node",
        IssueType.BUG: "issue_bug_subgraph_node",
        IssueType.FEATURE: END,
        IssueType.DOCUMENTATION: END,
        IssueType.QUESTION: "issue_question_subgraph_node",
      },
    )
    workflow.add_conditional_edges(
      "issue_classification_subgraph_node",
      lambda state: state["issue_type"],
      {
        IssueType.BUG: "issue_bug_subgraph_node",
        IssueType.FEATURE: END,
        IssueType.DOCUMENTATION: END,
        IssueType.QUESTION: "issue_question_subgraph_node",
      },
    )
    workflow.add_edge("issue_bug_subgraph_node", END)
    workflow.add_edge("issue_question_subgraph_node", END)

    self.graph = workflow.compile(checkpointer=checkpointer)

  def invoke(
    self,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    issue_type: IssueType,
  ):
    input_state = {
      "issue_title": issue_title,
      "issue_body": issue_body,
      "issue_comments": issue_comments,
      "issue_type": issue_type,
    }

    output_state = self.graph.invoke(input_state)
    return output_state
