import functools
from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.bug_fix_verification_subgraph_node import (
  BugFixVerificationSubgraphNode,
)
from prometheus.lang_graph.nodes.bug_reproduction_subgraph_node import BugReproductionSubgraphNode
from prometheus.lang_graph.nodes.build_and_test_subgraph_node import BuildAndTestSubgraphNode
from prometheus.lang_graph.nodes.context_provider_node import ContextProviderNode
from prometheus.lang_graph.nodes.edit_message_node import EditMessageNode
from prometheus.lang_graph.nodes.edit_node import EditNode
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.nodes.issue_bug_context_follow_up_message_node import (
  IssueBugContextFollowUpMessageNode,
)
from prometheus.lang_graph.nodes.issue_bug_context_message_node import IssueBugContextMessageNode
from prometheus.lang_graph.nodes.issue_bug_responder_node import IssueBugResponderNode
from prometheus.lang_graph.nodes.noop_node import NoopNode
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

    bug_reproduction_subgraph_node = BugReproductionSubgraphNode(
      model,
      container,
      kg,
      neo4j_driver,
      max_token_per_neo4j_result,
      test_commands,
      thread_id,
      checkpointer,
    )

    issue_bug_context_message_node = IssueBugContextMessageNode()
    context_provider_node = ContextProviderNode(model, kg, neo4j_driver, max_token_per_neo4j_result)
    context_provider_tools = ToolNode(
      tools=context_provider_node.tools,
      name="context_provider_tools",
      messages_key="context_provider_messages",
    )

    edit_message_node = EditMessageNode()
    edit_node = EditNode(model, kg)
    edit_tools = ToolNode(
      tools=edit_node.tools,
      name="edit_tools",
      messages_key="edit_messages",
    )
    git_diff_node = GitDiffNode(kg, exclude_reproduced_bug_file=True)
    update_container_node = UpdateContainerNode(self.container, kg)

    bug_fix_verification_branch_node = NoopNode()
    bug_fix_verification_subgraph_node = BugFixVerificationSubgraphNode(
      model, container, thread_id, checkpointer
    )
    build_or_test_branch_node = NoopNode()
    build_and_test_subgraph_node = BuildAndTestSubgraphNode(
      container, model, kg, build_commands, test_commands, thread_id, checkpointer
    )
    issue_bug_context_follow_up_message_node = IssueBugContextFollowUpMessageNode()
    issue_bug_responder_node = IssueBugResponderNode(model)

    workflow = StateGraph(IssueBugState)

    workflow.add_node("bug_reproduction_subgraph_node", bug_reproduction_subgraph_node)

    workflow.add_node("issue_bug_context_message_node", issue_bug_context_message_node)
    workflow.add_node("context_provider_node", context_provider_node)
    workflow.add_node("context_provider_tools", context_provider_tools)

    workflow.add_node("edit_message_node", edit_message_node)
    workflow.add_node("edit_node", edit_node)
    workflow.add_node("edit_tools", edit_tools)
    workflow.add_node("git_diff_node", git_diff_node)
    workflow.add_node("update_container_node", update_container_node)

    workflow.add_node("bug_fix_verification_branch_node", bug_fix_verification_branch_node)
    workflow.add_node("bug_fix_verification_subgraph_node", bug_fix_verification_subgraph_node)
    workflow.add_node("build_or_test_branch_node", build_or_test_branch_node)
    workflow.add_node("build_and_test_subgraph_node", build_and_test_subgraph_node)
    workflow.add_node(
      "issue_bug_context_follow_up_message_node", issue_bug_context_follow_up_message_node
    )
    workflow.add_node("issue_bug_responder_node", issue_bug_responder_node)

    workflow.set_entry_point("bug_reproduction_subgraph_node")
    workflow.add_edge("bug_reproduction_subgraph_node", "issue_bug_context_message_node")
    workflow.add_edge("issue_bug_context_message_node", "context_provider_node")
    workflow.add_conditional_edges(
      "context_provider_node",
      functools.partial(tools_condition, messages_key="context_provider_messages"),
      {"tools": "context_provider_tools", END: "edit_message_node"},
    )
    workflow.add_edge("context_provider_tools", "context_provider_node")

    workflow.add_edge("edit_message_node", "edit_node")
    workflow.add_conditional_edges(
      "edit_node",
      functools.partial(tools_condition, messages_key="edit_messages"),
      {"tools": "edit_tools", END: "git_diff_node"},
    )
    workflow.add_edge("edit_tools", "edit_node")
    workflow.add_edge("git_diff_node", "update_container_node")
    workflow.add_edge("update_container_node", "bug_fix_verification_branch_node")

    workflow.add_conditional_edges(
      "bug_fix_verification_branch_node",
      lambda state: state["reproduced_bug"],
      {True: "bug_fix_verification_subgraph_node", False: "build_or_test_branch_node"},
    )
    workflow.add_conditional_edges(
      "bug_fix_verification_subgraph_node",
      lambda state: state["reproducing_test_passed"],
      {True: "build_or_test_branch_node", False: "issue_bug_context_follow_up_message_node"},
    )
    workflow.add_conditional_edges(
      "build_or_test_branch_node",
      lambda state: state["run_build"] or state["run_existing_test"],
      {True: "build_and_test_subgraph_node", False: "issue_bug_responder_node"},
    )
    workflow.add_conditional_edges(
      "build_and_test_subgraph_node",
      lambda state: bool(state["build_fail_log"]) or bool(state["existing_test_fail_log"]),
      {True: "issue_bug_context_follow_up_message_node", False: "issue_bug_responder_node"},
    )
    workflow.add_edge("issue_bug_context_follow_up_message_node", "context_provider_node")
    workflow.add_edge("issue_bug_responder_node", END)

    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(
    self,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    run_build: bool,
    run_existing_test: bool,
    recursion_limit: int = 150,
  ):
    config = {"recursion_limit": recursion_limit}
    if self.thread_id:
      config["configurable"] = {"thread_id": self.thread_id}

    input_state = {
      "issue_title": issue_title,
      "issue_body": issue_body,
      "issue_comments": issue_comments,
      "run_build": run_build,
      "run_existing_test": run_existing_test,
    }

    try:
      if not self.container.is_running():
        self.container.build_docker_image()
        self.container.start_container()
      output_state = self.subgraph.invoke(input_state, config)
      return {
        "issue_response": output_state["issue_response"],
        "patch": output_state["patch"],
        "reproduced_bug_file": output_state.get("reproduced_bug_file", ""),
      }
    except GraphRecursionError:
      return {
        "issue_response": "",
        "patch": "",
        "reproduced_bug_file": "",
      }
    finally:
      self.container.cleanup()
