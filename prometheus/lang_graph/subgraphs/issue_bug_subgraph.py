import functools
from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueType
from prometheus.lang_graph.nodes.bug_fix_verification_subgraph_node import (
  BugFixVerificationSubgraphNode,
)
from prometheus.lang_graph.nodes.bug_fixing_node import BugFixingNode
from prometheus.lang_graph.nodes.bug_reproduction_subgraph_node import BugReproductionSubgraphNode
from prometheus.lang_graph.nodes.build_and_test_subgraph_node import BuildAndTestSubgraphNode
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.nodes.issue_to_context_node import IssueToContextNode
from prometheus.lang_graph.nodes.noop_node import NoopNode
from prometheus.lang_graph.nodes.reset_messages_node import ResetMessagesNode
from prometheus.lang_graph.nodes.update_container_node import UpdateContainerNode
from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState


class IssueBugSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.container = container
    self.thread_id = thread_id

    issue_to_bug_context_node = IssueToContextNode(
      IssueType.BUG, model, kg, neo4j_driver, max_token_per_neo4j_result, thread_id, checkpointer
    )
    bug_reproduction_subgraph_node = BugReproductionSubgraphNode(
      model, container, kg, test_commands, thread_id, checkpointer
    )
    bug_fixing_node = BugFixingNode(model, kg)
    bug_fixing_tools = ToolNode(
      tools=bug_fixing_node.tools,
      name="bug_fixing_tools",
      messages_key="bug_fixing_messages",
    )
    git_diff_node = GitDiffNode(kg)
    reset_bug_fixing_messages_node = ResetMessagesNode("bug_fixing_messages")
    update_container_node = UpdateContainerNode(self.container, kg)

    bug_fix_verification_branch_node = NoopNode()
    bug_fix_verification_subgraph_node = BugFixVerificationSubgraphNode(
      model, container, thread_id, checkpointer
    )
    build_or_test_branch_node = NoopNode()
    build_and_test_subgraph_node = BuildAndTestSubgraphNode(
      container, model, kg, build_commands, test_commands, thread_id, checkpointer
    )

    workflow = StateGraph(IssueBugState)

    workflow.add_node("issue_to_bug_context_node", issue_to_bug_context_node)

    workflow.add_node("bug_reproduction_subgraph_node", bug_reproduction_subgraph_node)

    workflow.add_node("bug_fixing_node", bug_fixing_node)
    workflow.add_node("bug_fixing_tools", bug_fixing_tools)
    workflow.add_node("git_diff_node", git_diff_node)
    workflow.add_node("reset_bug_fixing_messages_node", reset_bug_fixing_messages_node)
    workflow.add_node("update_container_node", update_container_node)

    workflow.add_node("bug_fix_verification_branch_node", bug_fix_verification_branch_node)
    workflow.add_node("bug_fix_verification_subgraph_node", bug_fix_verification_subgraph_node)
    workflow.add_node("build_or_test_branch_node", build_or_test_branch_node)
    workflow.add_node("build_and_test_subgraph_node", build_and_test_subgraph_node)

    workflow.set_entry_point("issue_to_bug_context_node")
    workflow.add_edge("issue_to_bug_context_node", "bug_reproduction_subgraph_node")
    workflow.add_edge("bug_reproduction_subgraph_node", "bug_fixing_node")
    workflow.add_conditional_edges(
      "bug_fixing_node",
      functools.partial(tools_condition, messages_key="bug_fixing_messages"),
      {"tools": "bug_fixing_tools", END: "git_diff_node"},
    )
    workflow.add_edge("bug_fixing_tools", "bug_fixing_node")
    workflow.add_edge("git_diff_node", "reset_bug_fixing_messages_node")
    workflow.add_edge("reset_bug_fixing_messages_node", "update_container_node")
    workflow.add_edge("update_container_node", "bug_fix_verification_branch_node")

    workflow.add_conditional_edges(
      "bug_fix_verification_branch_node",
      lambda state: state["reproduced_bug"],
      {True: "bug_fix_verification_subgraph_node", False: "build_or_test_branch_node"},
    )
    workflow.add_conditional_edges(
      "bug_fix_verification_subgraph_node",
      lambda state: state["fixed_bug"],
      {True: "build_or_test_branch_node", False: "bug_fixing_node"},
    )
    workflow.add_conditional_edges(
      "build_or_test_branch_node",
      lambda state: state["run_build"] or state["run_existing_test"],
      {True: "build_and_test_subgraph_node", False: END},
    )
    workflow.add_conditional_edges(
      "build_and_test_subgraph_node",
      lambda state: state["build_fail_log"] or state["existing_test_fail_log"],
      {True: "bug_fixing_node", False: END},
    )

    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(
    self,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    run_build: bool,
    run_existing_test: bool,
    recursion_limit: int = 300,
  ):
    config = {"recursion_limit": recursion_limit}
    if self.thread_id:
      config["configurable"] = {"thread_id": self.thread_id}

    if not self.container.is_running():
      self.container.build_docker_image()
      self.container.start_container()

    input_state = {
      "issue_title": issue_title,
      "issue_body": issue_body,
      "issue_comments": issue_comments,
      "run_build": run_build,
      "run_existing_test": run_existing_test,
    }

    output_state = self.subgraph.invoke(input_state, config)

    self.container.cleanup()
