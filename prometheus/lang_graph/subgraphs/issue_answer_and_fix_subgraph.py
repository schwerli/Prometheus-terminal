import functools
from pathlib import Path
from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.code_editing_node import CodeEditingNode
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.nodes.issue_responder_node import IssueResponderNode
from prometheus.lang_graph.nodes.issue_to_query_node import IssueToQueryNode
from prometheus.lang_graph.nodes.reset_edit_messages_node import ResetEditMessagesNode
from prometheus.lang_graph.nodes.run_test_node import RunTestNode
from prometheus.lang_graph.nodes.test_output_router import TestOutputRouter
from prometheus.lang_graph.subgraphs.context_provider_subgraph import ContextProviderSubgraph
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class IssueAnswerAndFixSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    local_path: Path,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.local_path = local_path.absolute()

    before_patch_test_node = RunTestNode("before_test_output")
    issue_to_query_node = IssueToQueryNode()
    context_provider_subgraph = ContextProviderSubgraph(
      model, kg, neo4j_driver, checkpointer
    ).subgraph
    code_editing_node = CodeEditingNode(model, str(self.local_path))
    code_editing_tools = ToolNode(
      tools=code_editing_node.tools, name="code_editing_tools", messages_key="code_edit_messages"
    )
    after_patch_test_node = RunTestNode("after_test_output")
    git_diff_node = GitDiffNode()
    issue_responder_node = IssueResponderNode(model)
    reset_edit_messages_node = ResetEditMessagesNode()

    workflow = StateGraph(IssueAnswerAndFixState)
    workflow.add_node("before_patch_test_node", before_patch_test_node)
    workflow.add_node("issue_to_query_node", issue_to_query_node)
    workflow.add_node("context_provider_subgraph", context_provider_subgraph)
    workflow.add_node("code_editing_node", code_editing_node)
    workflow.add_node("code_editing_tools", code_editing_tools)
    workflow.add_node("after_patch_test_node", after_patch_test_node)
    workflow.add_node("git_diff_node", git_diff_node)
    workflow.add_node("issue_responder_node", issue_responder_node)
    workflow.add_node("reset_edit_messages_node", reset_edit_messages_node)

    workflow.add_edge("before_patch_test_node", "issue_to_query_node")
    workflow.add_edge("issue_to_query_node", "context_provider_subgraph")
    workflow.add_edge("context_provider_subgraph", "code_editing_node")
    workflow.add_conditional_edges(
      "code_editing_node",
      functools.partial(tools_condition, messages_key="code_edit_messages"),
      {"tools": "code_editing_tools", END: "git_diff_node"},
    )
    workflow.add_edge("code_editing_tools", "code_editing_node")
    workflow.add_edge("git_diff_node", "after_patch_test_node")
    workflow.add_conditional_edges(
      "after_patch_test_node",
      TestOutputRouter(model),
      {True: "issue_responder_node", False: "reset_edit_messages_node"},
    )
    workflow.add_edge("reset_edit_messages_node", "code_editing_node")
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
      {
        "project_path": str(self.local_path),
        "issue_title": issue_title,
        "issue_body": issue_body,
        "issue_comments": issue_comments,
      },
      config,
    )
    return output_state["issue_response"].content
