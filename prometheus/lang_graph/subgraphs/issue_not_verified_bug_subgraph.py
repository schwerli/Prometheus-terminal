import functools
from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.context_provider_node import ContextProviderNode
from prometheus.lang_graph.nodes.edit_message_node import EditMessageNode
from prometheus.lang_graph.nodes.edit_node import EditNode
from prometheus.lang_graph.nodes.final_patch_selection_node import FinalPatchSelectionNode
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.nodes.git_reset_node import GitResetNode
from prometheus.lang_graph.nodes.issue_bug_analyzer_message_node import IssueBugAnalyzerMessageNode
from prometheus.lang_graph.nodes.issue_bug_analyzer_node import IssueBugAnalyzerNode
from prometheus.lang_graph.nodes.issue_bug_context_message_node import IssueBugContextMessageNode
from prometheus.lang_graph.nodes.reset_messages_node import ResetMessagesNode
from prometheus.lang_graph.subgraphs.issue_not_verified_bug_state import IssueNotVerifiedBugState


class IssueNotVerifiedBugSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    git_repo: GitRepository,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.thread_id = thread_id

    issue_bug_context_message_node = IssueBugContextMessageNode()
    context_provider_node = ContextProviderNode(model, kg, neo4j_driver, max_token_per_neo4j_result)
    context_provider_tools = ToolNode(
      tools=context_provider_node.tools,
      name="context_provider_tools",
      messages_key="context_provider_messages",
    )

    issue_bug_analyzer_message_node = IssueBugAnalyzerMessageNode()
    issue_bug_analyzer_node = IssueBugAnalyzerNode(model)

    edit_message_node = EditMessageNode()
    edit_node = EditNode(model, kg)
    edit_tools = ToolNode(
      tools=edit_node.tools,
      name="edit_tools",
      messages_key="edit_messages",
    )
    git_diff_node = GitDiffNode(git_repo, "edit_patches", return_list=True)

    git_reset_node = GitResetNode(git_repo)
    reset_context_provider_messages_node = ResetMessagesNode("context_provider_messages")
    reset_issue_bug_analyzer_messages_node = ResetMessagesNode("issue_bug_analyzer_messages")
    reset_edit_messages_node = ResetMessagesNode("edit_messages")

    final_patch_selection_node = FinalPatchSelectionNode(model)

    workflow = StateGraph(IssueNotVerifiedBugState)

    workflow.add_node("issue_bug_context_message_node", issue_bug_context_message_node)
    workflow.add_node("context_provider_node", context_provider_node)
    workflow.add_node("context_provider_tools", context_provider_tools)

    workflow.add_node("issue_bug_analyzer_message_node", issue_bug_analyzer_message_node)
    workflow.add_node("issue_bug_analyzer_node", issue_bug_analyzer_node)

    workflow.add_node("edit_message_node", edit_message_node)
    workflow.add_node("edit_node", edit_node)
    workflow.add_node("edit_tools", edit_tools)
    workflow.add_node("git_diff_node", git_diff_node)

    workflow.add_node("git_reset_node", git_reset_node)
    workflow.add_node("reset_context_provider_messages_node", reset_context_provider_messages_node)
    workflow.add_node(
      "reset_issue_bug_analyzer_messages_node", reset_issue_bug_analyzer_messages_node
    )
    workflow.add_node("reset_edit_messages_node", reset_edit_messages_node)

    workflow.add_node("final_patch_selection_node", final_patch_selection_node)

    workflow.set_entry_point("issue_bug_context_message_node")
    workflow.add_edge("issue_bug_context_message_node", "context_provider_node")
    workflow.add_conditional_edges(
      "context_provider_node",
      functools.partial(tools_condition, messages_key="context_provider_messages"),
      {"tools": "context_provider_tools", END: "issue_bug_analyzer_message_node"},
    )
    workflow.add_edge("context_provider_tools", "context_provider_node")

    workflow.add_edge("issue_bug_analyzer_message_node", "issue_bug_analyzer_node")
    workflow.add_edge("issue_bug_analyzer_node", "edit_message_node")

    workflow.add_edge("edit_message_node", "edit_node")
    workflow.add_conditional_edges(
      "edit_node",
      functools.partial(tools_condition, messages_key="edit_messages"),
      {"tools": "edit_tools", END: "git_diff_node"},
    )
    workflow.add_edge("edit_tools", "edit_node")

    workflow.add_conditional_edges(
      "git_diff_node",
      lambda state: len(state["edit_patches"]) < state["number_of_candidate_patch"],
      {True: "git_reset_node", False: "final_patch_selection_node"},
    )

    workflow.add_edge("git_reset_node", "reset_context_provider_messages_node")
    workflow.add_edge(
      "reset_context_provider_messages_node", "reset_issue_bug_analyzer_messages_node"
    )
    workflow.add_edge("reset_issue_bug_analyzer_messages_node", "reset_edit_messages_node")
    workflow.add_edge("reset_edit_messages_node", "issue_bug_context_message_node")

    workflow.add_edge("final_patch_selection_node", END)

    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(
    self,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    number_of_candidate_patch: int,
    recursion_limit: int = 999,
  ):
    config = {"recursion_limit": recursion_limit}
    if self.thread_id:
      config["configurable"] = {"thread_id": self.thread_id}

    input_state = {
      "issue_title": issue_title,
      "issue_body": issue_body,
      "issue_comments": issue_comments,
      "number_of_candidate_patch": number_of_candidate_patch,
    }

    output_state = self.subgraph.invoke(input_state, config)
    return {
      "final_patch": output_state["final_patch"],
    }
