import functools
from pathlib import Path
from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.docker.general_container import GeneralContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.code_editing_node import CodeEditingNode
from prometheus.lang_graph.nodes.general_build_node import GeneralBuildNode
from prometheus.lang_graph.nodes.general_build_summarize_node import GeneralBuildSummarizeNode
from prometheus.lang_graph.nodes.general_test_node import GeneralTestNode
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.nodes.issue_responder_node import IssueResponderNode
from prometheus.lang_graph.nodes.issue_to_query_node import IssueToQueryNode
from prometheus.lang_graph.nodes.noop_node import NoopNode
from prometheus.lang_graph.nodes.reset_messages_node import ResetMessagesNode
from prometheus.lang_graph.routers.issue_answer_and_fix_need_build_router import (
  IssueAnswerAndFixNeedBuildRouter,
)
from prometheus.lang_graph.routers.issue_answer_and_fix_need_test_router import (
  IssueAnswerAndFixNeedTestRouter,
)
from prometheus.lang_graph.routers.issue_answer_and_fix_success_router import (
  IssueAnswerAndFixSuccessRouter,
)
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
    general_container = GeneralContainer(self.local_path)

    issue_to_query_node = IssueToQueryNode()
    context_provider_subgraph = ContextProviderSubgraph(
      model, kg, neo4j_driver, checkpointer
    ).subgraph

    before_edit_build_branch_node = NoopNode()
    before_edit_general_build_node = GeneralBuildNode(model, general_container, before_edit=True)
    before_edit_general_build_tools = ToolNode(tools=before_edit_general_build_node.tools, name="before_edit_general_build_tools", messages_key="build_messages")
    before_edit_general_build_summarize_node = GeneralBuildSummarizeNode(model)
    before_edit_test_branch_node = NoopNode()
    before_edit_general_test_node = GeneralTestNode(model, general_container, before_edit=True)
    before_edit_general_test_tools = ToolNode(tools=before_edit_general_test_node.tools, name="before_edit_general_test_tools", messages_key="test_messages")
    before_general_test_summarization_node = GeneralBuildSummarizeNode(model)

    code_editing_node = CodeEditingNode(model, str(self.local_path))
    code_editing_tools = ToolNode(tools=code_editing_node.tools, name="code_editing_tools", messages_key="code_edit_messages")
    git_diff_node = GitDiffNode()
    reset_edit_messages_node = ResetMessagesNode("code_edit_messages")
    reset_build_messages_node = ResetMessagesNode("build_messages")
    reset_test_messages_node = ResetMessagesNode("test_messages")

    after_edit_build_branch_node = NoopNode()
    after_edit_general_build_node = GeneralBuildNode(model, general_container, before_edit=False)
    after_edit_general_build_tools = ToolNode(tools=after_edit_general_build_node.tools, name="after_edit_general_build_tools", messages_key="build_messages")
    after_edit_general_build_summarize_node = GeneralBuildSummarizeNode(model)
    after_edit_test_branch_node = NoopNode()
    after_edit_general_test_node = GeneralTestNode(model, general_container, before_edit=False)
    after_edit_general_test_tools = ToolNode(tools=after_edit_general_test_node.tools, name="after_edit_general_test_tools", messages_key="test_messages")
    after_general_test_summarization_node = GeneralBuildSummarizeNode(model)

    issue_responder_node = IssueResponderNode(model)

    workflow = StateGraph(IssueAnswerAndFixState)
    workflow.add_node("issue_to_query_node", issue_to_query_node)
    workflow.add_node("context_provider_subgraph", context_provider_subgraph)
  
    workflow.add_node("before_edit_build_branch_node", before_edit_build_branch_node)
    workflow.add_node("before_edit_general_build_node", before_edit_general_build_node)
    workflow.add_node("before_edit_general_build_tools", before_edit_general_build_tools)
    workflow.add_node("before_edit_general_build_summarize_node", before_edit_general_build_summarize_node)
    workflow.add_node("before_edit_test_branch_node", before_edit_test_branch_node)
    workflow.add_node("before_edit_general_test_node", before_edit_general_test_node)
    workflow.add_node("before_edit_general_test_tools", before_edit_general_test_tools)
    workflow.add_node("before_general_test_summarization_node", before_general_test_summarization_node)

    workflow.add_node("code_editing_node", code_editing_node)
    workflow.add_node("code_editing_tools", code_editing_tools)
    workflow.add_node("git_diff_node", git_diff_node)
    workflow.add_node("reset_edit_messages_node", reset_edit_messages_node)
    workflow.add_node("reset_build_messages_node", reset_build_messages_node)
    workflow.add_node("reset_test_messages_node", reset_test_messages_node)

    workflow.add_node("after_edit_build_branch_node", after_edit_build_branch_node)
    workflow.add_node("after_edit_general_build_node", after_edit_general_build_node)
    workflow.add_node("after_edit_general_build_tools", after_edit_general_build_tools)
    workflow.add_node("after_edit_general_build_summarize_node", after_edit_general_build_summarize_node)
    workflow.add_node("after_edit_test_branch_node", after_edit_test_branch_node)
    workflow.add_node("after_edit_general_test_node", after_edit_general_test_node)
    workflow.add_node("after_edit_general_test_tools", after_edit_general_test_tools)
    workflow.add_node("after_general_test_summarization_node", after_general_test_summarization_node)

    workflow.add_node("issue_responder_node", issue_responder_node)

    workflow.add_edge("issue_to_query_node", "context_provider_subgraph")
    workflow.add_edge("context_provider_subgraph", "before_edit_build_branch_node")

    workflow.add_conditional_edges(
      "before_edit_build_branch_node",
      IssueAnswerAndFixNeedBuildRouter(),
      {True: "before_edit_general_build_node", False: "before_edit_test_branch_node"},
    )
    workflow.add_conditional_edges(
      "before_edit_general_build_node",
      functools.partial(tools_condition, messages_key="build_messages"),
      {"tools": "before_edit_general_build_tools", END: "before_edit_general_build_summarize_node"},
    )
    workflow.add_edge("before_edit_general_build_tools", "before_edit_general_build_node")
    workflow.add_edge("before_edit_general_build_summarize_node", "before_edit_test_branch_node")
    workflow.add_conditional_edges(
      "before_edit_test_branch_node",
      IssueAnswerAndFixNeedTestRouter(),
      {True: "before_edit_general_test_node", False: "code_editing_node"},
    )
    workflow.add_conditional_edges(
      "before_edit_general_test_node",
      functools.partial(tools_condition, messages_key="test_messages"),
      {"tools": "before_edit_general_test_tools", END: "before_edit_general_test_summarize_node"},
    )
    workflow.add_edge("before_edit_general_test_tools", "before_edit_general_test_node")
    workflow.add_edge("before_edit_general_test_summarize_node", "code_editing_node")


    workflow.add_conditional_edges(
      "code_editing_node",
      functools.partial(tools_condition, messages_key="code_edit_messages"),
      {"tools": "code_editing_tools", END: "git_diff_node"},
    )
    workflow.add_edge("code_editing_tools", "code_editing_node")
    workflow.add_edge("git_diff_node", "reset_edit_messages_node")
    workflow.add_edge("reset_edit_messages_node", "reset_build_messages_node")
    workflow.add_edge("reset_build_messages_node", "reset_test_messages_node")
    workflow.add_edge("reset_test_messages_node", "after_edit_build_branch_node")

    workflow.add_conditional_edges(
      "after_edit_build_branch_node",
      IssueAnswerAndFixNeedBuildRouter(),
      {True: "after_edit_general_build_node", False: "after_edit_test_branch_node"},
    )
    workflow.add_conditional_edges(
      "after_edit_general_build_node",
      functools.partial(tools_condition, messages_key="build_messages"),
      {"tools": "after_edit_general_build_tools", END: "after_edit_general_build_summarize_node"},
    )
    workflow.add_edge("after_edit_general_build_tools", "after_edit_general_build_node")
    workflow.add_conditional_edges(
      "after_edit_general_build_summarize_node",
      IssueAnswerAndFixSuccessRouter(),
      {True: "after_edit_test_branch_node", False: "code_editing_node"})
    workflow.add_conditional_edges(
      "after_edit_test_branch_node",
      IssueAnswerAndFixNeedTestRouter(),
      {True: "after_edit_general_test_node", False: "issue_responder_node"},
    )
    workflow.add_conditional_edges(
      "after_edit_general_test_node",
      functools.partial(tools_condition, messages_key="test_messages"),
      {"tools": "after_edit_general_test_tools", END: "after_edit_general_test_summarize_node"},
    )
    workflow.add_edge("after_edit_general_test_tools", "after_edit_general_test_node")
    workflow.add_conditional_edges(
      "after_edit_general_test_summarize_node",
      IssueAnswerAndFixSuccessRouter(),
      {True: "issue_responder_node", False: "code_editing_node"})

    workflow.add_edge("issue_responder_node", END)

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
