import functools
from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.docker.base_container import BaseContainer
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.bug_fix_verification_subgraph_node import (
  BugFixVerificationSubgraphNode,
)
from prometheus.lang_graph.nodes.build_and_test_subgraph_node import BuildAndTestSubgraphNode
from prometheus.lang_graph.nodes.context_retrieval_subgraph_node import ContextRetrievalSubgraphNode
from prometheus.lang_graph.nodes.edit_message_node import EditMessageNode
from prometheus.lang_graph.nodes.edit_node import EditNode
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.nodes.issue_bug_analyzer_message_node import IssueBugAnalyzerMessageNode
from prometheus.lang_graph.nodes.issue_bug_analyzer_node import IssueBugAnalyzerNode
from prometheus.lang_graph.nodes.issue_bug_context_message_node import IssueBugContextMessageNode
from prometheus.lang_graph.nodes.noop_node import NoopNode
from prometheus.lang_graph.nodes.update_container_node import UpdateContainerNode
from prometheus.lang_graph.subgraphs.issue_verified_bug_state import IssueVerifiedBugState


class IssueVerifiedBugSubgraph:
  def __init__(
    self,
    advanced_model: BaseChatModel,
    base_model: BaseChatModel,
    container: BaseContainer,
    kg: KnowledgeGraph,
    git_repo: GitRepository,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
  ):
    issue_bug_context_message_node = IssueBugContextMessageNode()
    context_retrieval_subgraph_node = ContextRetrievalSubgraphNode(
      model=base_model,
      kg=kg,
      neo4j_driver=neo4j_driver,
      max_token_per_neo4j_result=max_token_per_neo4j_result,
      query_key_name="bug_fix_query",
      context_key_name="bug_fix_context",
    )

    issue_bug_analyzer_message_node = IssueBugAnalyzerMessageNode()
    issue_bug_analyzer_node = IssueBugAnalyzerNode(advanced_model)

    edit_message_node = EditMessageNode()
    edit_node = EditNode(advanced_model, kg)
    edit_tools = ToolNode(
      tools=edit_node.tools,
      name="edit_tools",
      messages_key="edit_messages",
    )
    git_diff_node = GitDiffNode(git_repo, "edit_patch", "reproduced_bug_file")
    update_container_node = UpdateContainerNode(container, git_repo)

    bug_fix_verification_subgraph_node = BugFixVerificationSubgraphNode(
      base_model,
      container,
    )
    build_or_test_branch_node = NoopNode()
    build_and_test_subgraph_node = BuildAndTestSubgraphNode(
      container,
      advanced_model,
      kg,
      build_commands,
      test_commands,
    )

    workflow = StateGraph(IssueVerifiedBugState)

    workflow.add_node("issue_bug_context_message_node", issue_bug_context_message_node)
    workflow.add_node("context_retrieval_subgraph_node", context_retrieval_subgraph_node)

    workflow.add_node("issue_bug_analyzer_message_node", issue_bug_analyzer_message_node)
    workflow.add_node("issue_bug_analyzer_node", issue_bug_analyzer_node)

    workflow.add_node("edit_message_node", edit_message_node)
    workflow.add_node("edit_node", edit_node)
    workflow.add_node("edit_tools", edit_tools)
    workflow.add_node("git_diff_node", git_diff_node)
    workflow.add_node("update_container_node", update_container_node)

    workflow.add_node("bug_fix_verification_subgraph_node", bug_fix_verification_subgraph_node)
    workflow.add_node("build_or_test_branch_node", build_or_test_branch_node)
    workflow.add_node("build_and_test_subgraph_node", build_and_test_subgraph_node)

    workflow.set_entry_point("issue_bug_context_message_node")
    workflow.add_edge("issue_bug_context_message_node", "context_retrieval_subgraph_node")
    workflow.add_edge("context_retrieval_subgraph_node", "issue_bug_analyzer_message_node")

    workflow.add_edge("issue_bug_analyzer_message_node", "issue_bug_analyzer_node")
    workflow.add_edge("issue_bug_analyzer_node", "edit_message_node")

    workflow.add_edge("edit_message_node", "edit_node")
    workflow.add_conditional_edges(
      "edit_node",
      functools.partial(tools_condition, messages_key="edit_messages"),
      {"tools": "edit_tools", END: "git_diff_node"},
    )
    workflow.add_edge("edit_tools", "edit_node")
    workflow.add_edge("git_diff_node", "update_container_node")
    workflow.add_edge("update_container_node", "bug_fix_verification_subgraph_node")

    workflow.add_conditional_edges(
      "bug_fix_verification_subgraph_node",
      lambda state: bool(state["reproducing_test_fail_log"]),
      {True: "issue_bug_analyzer_message_node", False: "build_or_test_branch_node"},
    )
    workflow.add_conditional_edges(
      "build_or_test_branch_node",
      lambda state: state["run_build"] or state["run_existing_test"],
      {True: "build_and_test_subgraph_node", False: END},
    )
    workflow.add_conditional_edges(
      "build_and_test_subgraph_node",
      lambda state: bool(state["build_fail_log"]) or bool(state["existing_test_fail_log"]),
      {True: "issue_bug_analyzer_message_node", False: END},
    )

    self.subgraph = workflow.compile()

  def invoke(
    self,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    run_build: bool,
    run_existing_test: bool,
    reproduced_bug_file: str,
    reproduced_bug_commands: Sequence[str],
    recursion_limit: int = 80,
  ):
    config = {"recursion_limit": recursion_limit}

    input_state = {
      "issue_title": issue_title,
      "issue_body": issue_body,
      "issue_comments": issue_comments,
      "run_build": run_build,
      "run_existing_test": run_existing_test,
      "reproduced_bug_file": reproduced_bug_file,
      "reproduced_bug_commands": reproduced_bug_commands,
      "max_refined_query_loop": 3,
    }

    output_state = self.subgraph.invoke(input_state, config)
    return {
      "edit_patch": output_state["edit_patch"],
      "reproducing_test_fail_log": output_state["reproducing_test_fail_log"],
      "exist_build": output_state.get("exist_build", False),
      "build_fail_log": output_state.get("build_fail_log", ""),
      "exist_test": output_state.get("exist_test", False),
      "existing_test_fail_log": output_state.get("existing_test_fail_log", ""),
    }
