from pathlib import Path
from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.code_editing_node import CodeEditingNode
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.nodes.issue_responder_node import IssueResponderNode
from prometheus.lang_graph.nodes.issue_to_query_node import IssueToQueryNode
from prometheus.lang_graph.nodes.run_test_node import RunTestNode
from prometheus.lang_graph.nodes.test_output_router import TestOutputRouter
from prometheus.lang_graph.subgraphs.context_provider_subgraph import ContextProviderSubgraph
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueAnswerAndFixSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    project_path: Path,
    git_repo: GitRepository,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    before_patch_test_node = RunTestNode(project_path, "before_test_output")
    issue_to_query_node = IssueToQueryNode()
    context_provider_subgraph = ContextProviderSubgraph(
      model, kg, neo4j_driver, checkpointer
    ).subgraph
    code_editing_node = CodeEditingNode(model, project_path)
    after_patch_test_node = RunTestNode(project_path, "after_test_output")
    git_diff_node = GitDiffNode(git_repo)
    issue_responder_node = IssueResponderNode(model)

    workflow = StateGraph(IssueAnswerAndFixState)
    workflow.add_node("before_patch_test_node", before_patch_test_node)
    workflow.add_node("issue_to_query_node", issue_to_query_node)
    workflow.add_node("context_provider_subgraph", context_provider_subgraph)
    workflow.add_node("code_editing_node", code_editing_node)
    workflow.add_node("after_patch_test_node", after_patch_test_node)
    workflow.add_node("git_diff_node", git_diff_node)
    workflow.add_node("issue_responder_node", issue_responder_node)

    workflow.add_edge("before_patch_test_node", "issue_to_query_node")
    workflow.add_edge("issue_to_query_node", "context_provider_subgraph")
    workflow.add_edge("context_provider_subgraph", "code_editing_node")
    workflow.add_edge("code_editing_node", "after_patch_test_node")
    workflow.add_conditional_edges(
      "after_patch_test_node",
      TestOutputRouter(model),
      {True: "git_diff_node", False: "code_editing_node"},
    )
    workflow.add_edge("git_diff_node", "issue_responder_node")
    workflow.add_edge("issue_responder_node", END)
    workflow.set_entry_point("before_patch_test_node")
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
    return output_state["issue_response"].content
