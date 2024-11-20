import functools
from pathlib import Path
from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.docker.general_container import GeneralContainer
from prometheus.docker.user_defined_container import UserDefinedContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.code_editing_node import CodeEditingNode
from prometheus.lang_graph.nodes.edit_reviewer_node import EditReviewerNode
from prometheus.lang_graph.nodes.edit_reviewer_structured_node import EditReviewerStructuredNode
from prometheus.lang_graph.nodes.general_build_node import GeneralBuildNode
from prometheus.lang_graph.nodes.general_build_structured_node import (
  GeneralBuildStructuredNode,
)
from prometheus.lang_graph.nodes.general_test_node import GeneralTestNode
from prometheus.lang_graph.nodes.general_test_structured_node import GeneralTestStructuredNode
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.nodes.issue_responder_node import IssueResponderNode
from prometheus.lang_graph.nodes.issue_to_context_node import IssueToQueryNode
from prometheus.lang_graph.nodes.noop_node import NoopNode
from prometheus.lang_graph.nodes.require_edit_classifier_node import RequireEditClassifierNode
from prometheus.lang_graph.nodes.reset_messages_node import ResetMessagesNode
from prometheus.lang_graph.nodes.update_container_node import UpdateContainerNode
from prometheus.lang_graph.nodes.user_defined_build_node import UserDefinedBuildNode
from prometheus.lang_graph.nodes.user_defined_test_node import UserDefinedTestNode
from prometheus.lang_graph.routers.issue_answer_and_fix_need_build_router import (
  IssueAnswerAndFixNeedBuildRouter,
)
from prometheus.lang_graph.routers.issue_answer_and_fix_need_test_router import (
  IssueAnswerAndFixNeedTestRouter,
)
from prometheus.lang_graph.routers.issue_answer_and_fix_only_answer_router import (
  IssueAnswerAndFixOnlyAnswerRouter,
)
from prometheus.lang_graph.routers.issue_answer_and_fix_success_router import (
  IssueAnswerAndFixSuccessRouter,
)
from prometheus.lang_graph.subgraphs.context_provider_subgraph import ContextProviderSubgraph
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import (
  IssueAnswerAndFixState,
  ResponseModeEnum,
)


class IssueAnswerAndFixSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    local_path: Path,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    dockerfile_content: Optional[str] = None,
    image_name: Optional[str] = None,
    workdir: Optional[str] = None,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
  ):
    self.local_path = local_path.absolute()
    self.project_structure = kg.get_file_tree()

    is_using_user_defined_container = bool(dockerfile_content) or bool(image_name)

    if is_using_user_defined_container:
      self.container = UserDefinedContainer(
        self.local_path, build_commands, test_commands, workdir, dockerfile_content, image_name
      )
    else:
      self.container = GeneralContainer(self.local_path)

    issue_to_query_node = IssueToQueryNode()
    context_provider_subgraph = ContextProviderSubgraph(
      model, kg, neo4j_driver, max_token_per_neo4j_result, checkpointer
    ).subgraph

    require_edit_classifier_node = RequireEditClassifierNode(model)

    before_edit_build_branch_node = NoopNode()
    if is_using_user_defined_container:
      before_edit_build_node = UserDefinedBuildNode(self.container)
    else:
      before_edit_build_node = GeneralBuildNode(model, self.container, before_edit=True)
      before_edit_build_tools = ToolNode(
        tools=before_edit_build_node.tools,
        name="before_edit_build_tools",
        messages_key="build_messages",
      )
    before_edit_general_build_structured_node = GeneralBuildStructuredNode(model)
    before_edit_test_branch_node = NoopNode()
    if is_using_user_defined_container:
      before_edit_test_node = UserDefinedTestNode(self.container)
    else:
      before_edit_test_node = GeneralTestNode(model, self.container, before_edit=True)
      before_edit_test_tools = ToolNode(
        tools=before_edit_test_node.tools,
        name="before_edit_test_tools",
        messages_key="test_messages",
      )
    before_edit_general_test_structured_node = GeneralTestStructuredNode(model)

    code_editing_node = CodeEditingNode(model, str(self.local_path))
    code_editing_tools = ToolNode(
      tools=code_editing_node.tools, name="code_editing_tools", messages_key="code_edit_messages"
    )
    git_diff_node = GitDiffNode()
    reset_edit_messages_node = ResetMessagesNode("code_edit_messages")
    reset_reviewer_messages_node = ResetMessagesNode("edit_reviewer_messages")
    reset_build_messages_node = ResetMessagesNode("build_messages")
    reset_test_messages_node = ResetMessagesNode("test_messages")
    reset_build_fail_log_node = ResetMessagesNode("build_fail_log")
    reset_test_fail_log_node = ResetMessagesNode("test_fail_log")
    update_container_node = UpdateContainerNode(self.container)

    edit_reviewer_node = EditReviewerNode(model, str(self.local_path))
    edit_reviewer_tools = ToolNode(
      tools=edit_reviewer_node.tools,
      name="edit_reviewer_tools",
      messages_key="edit_reviewer_messages",
    )
    edit_reviewer_structured_node = EditReviewerStructuredNode(model)

    after_edit_build_branch_node = NoopNode()
    if is_using_user_defined_container:
      after_edit_build_node = UserDefinedBuildNode(self.container)
    else:
      after_edit_build_node = GeneralBuildNode(model, self.container, before_edit=False)
      after_edit_build_tools = ToolNode(
        tools=after_edit_build_node.tools,
        name="after_edit_build_tools",
        messages_key="build_messages",
      )
    after_edit_general_build_structured_node = GeneralBuildStructuredNode(model)
    after_edit_test_branch_node = NoopNode()
    if is_using_user_defined_container:
      after_edit_test_node = UserDefinedTestNode(self.container)
    else:
      after_edit_test_node = GeneralTestNode(model, self.container, before_edit=False)
      after_edit_test_tools = ToolNode(
        tools=after_edit_test_node.tools,
        name="after_edit_test_tools",
        messages_key="test_messages",
      )
    after_edit_general_test_structured_node = GeneralTestStructuredNode(model)

    issue_responder_node = IssueResponderNode(model)

    workflow = StateGraph(IssueAnswerAndFixState)
    workflow.add_node("issue_to_query_node", issue_to_query_node)
    workflow.add_node("context_provider_subgraph", context_provider_subgraph)

    workflow.add_node("require_edit_classifier_node", require_edit_classifier_node)

    workflow.add_node("before_edit_build_branch_node", before_edit_build_branch_node)
    workflow.add_node("before_edit_build_node", before_edit_build_node)
    if not dockerfile_content:
      workflow.add_node("before_edit_build_tools", before_edit_build_tools)
    workflow.add_node(
      "before_edit_general_build_structured_node", before_edit_general_build_structured_node
    )
    workflow.add_node("before_edit_test_branch_node", before_edit_test_branch_node)
    workflow.add_node("before_edit_test_node", before_edit_test_node)
    if not dockerfile_content:
      workflow.add_node("before_edit_test_tools", before_edit_test_tools)
    workflow.add_node(
      "before_edit_general_test_structured_node", before_edit_general_test_structured_node
    )

    workflow.add_node("code_editing_node", code_editing_node)
    workflow.add_node("code_editing_tools", code_editing_tools)
    workflow.add_node("git_diff_node", git_diff_node)

    workflow.add_node("reset_edit_messages_node", reset_edit_messages_node)
    workflow.add_node("reset_reviewer_messages_node", reset_reviewer_messages_node)
    workflow.add_node("reset_build_messages_node", reset_build_messages_node)
    workflow.add_node("reset_test_messages_node", reset_test_messages_node)
    workflow.add_node("reset_build_fail_log_node", reset_build_fail_log_node)
    workflow.add_node("reset_test_fail_log_node", reset_test_fail_log_node)
    workflow.add_node("update_container_node", update_container_node)

    workflow.add_node("edit_reviewer_node", edit_reviewer_node)
    workflow.add_node("edit_reviewer_tools", edit_reviewer_tools)
    workflow.add_node("edit_reviewer_structured_node", edit_reviewer_structured_node)

    workflow.add_node("after_edit_build_branch_node", after_edit_build_branch_node)
    workflow.add_node("after_edit_build_node", after_edit_build_node)
    if not dockerfile_content:
      workflow.add_node("after_edit_build_tools", after_edit_build_tools)
    workflow.add_node(
      "after_edit_general_build_structured_node", after_edit_general_build_structured_node
    )
    workflow.add_node("after_edit_test_branch_node", after_edit_test_branch_node)
    workflow.add_node("after_edit_test_node", after_edit_test_node)
    if not dockerfile_content:
      workflow.add_node("after_edit_test_tools", after_edit_test_tools)
    workflow.add_node(
      "after_edit_general_test_structured_node", after_edit_general_test_structured_node
    )

    workflow.add_node("issue_responder_node", issue_responder_node)

    workflow.add_edge("issue_to_query_node", "context_provider_subgraph")
    workflow.add_conditional_edges(
      "context_provider_subgraph",
      IssueAnswerAndFixOnlyAnswerRouter(),
      {True: "issue_responder_node", False: "require_edit_classifier_node"},
    )

    workflow.add_conditional_edges(
      "require_edit_classifier_node",
      IssueAnswerAndFixOnlyAnswerRouter(),
      {True: "issue_responder_node", False: "before_edit_build_branch_node"},
    )

    workflow.add_conditional_edges(
      "before_edit_build_branch_node",
      IssueAnswerAndFixNeedBuildRouter(),
      {True: "before_edit_build_node", False: "before_edit_test_branch_node"},
    )
    if is_using_user_defined_container:
      workflow.add_edge("before_edit_build_node", "before_edit_general_build_structured_node")
    else:
      workflow.add_conditional_edges(
        "before_edit_build_node",
        functools.partial(tools_condition, messages_key="build_messages"),
        {
          "tools": "before_edit_build_tools",
          END: "before_edit_general_build_structured_node",
        },
      )
      workflow.add_edge("before_edit_build_tools", "before_edit_build_node")
    workflow.add_edge("before_edit_general_build_structured_node", "before_edit_test_branch_node")
    workflow.add_conditional_edges(
      "before_edit_test_branch_node",
      IssueAnswerAndFixNeedTestRouter(),
      {True: "before_edit_test_node", False: "code_editing_node"},
    )
    if is_using_user_defined_container:
      workflow.add_edge("before_edit_test_node", "before_edit_general_test_structured_node")
    else:
      workflow.add_conditional_edges(
        "before_edit_test_node",
        functools.partial(tools_condition, messages_key="test_messages"),
        {
          "tools": "before_edit_test_tools",
          END: "before_edit_general_test_structured_node",
        },
      )
      workflow.add_edge("before_edit_test_tools", "before_edit_test_node")
    workflow.add_edge("before_edit_general_test_structured_node", "code_editing_node")

    workflow.add_conditional_edges(
      "code_editing_node",
      functools.partial(tools_condition, messages_key="code_edit_messages"),
      {"tools": "code_editing_tools", END: "git_diff_node"},
    )
    workflow.add_edge("code_editing_tools", "code_editing_node")
    workflow.add_edge("git_diff_node", "reset_edit_messages_node")
    workflow.add_edge("reset_edit_messages_node", "reset_reviewer_messages_node")
    workflow.add_edge("reset_reviewer_messages_node", "reset_build_messages_node")
    workflow.add_edge("reset_build_messages_node", "reset_test_messages_node")
    workflow.add_edge("reset_test_messages_node", "reset_build_fail_log_node")
    workflow.add_edge("reset_build_fail_log_node", "reset_test_fail_log_node")
    workflow.add_edge("reset_test_fail_log_node", "update_container_node")
    workflow.add_edge("update_container_node", "edit_reviewer_node")

    workflow.add_conditional_edges(
      "edit_reviewer_node",
      functools.partial(tools_condition, messages_key="edit_reviewer_messages"),
      {"tools": "edit_reviewer_tools", END: "edit_reviewer_structured_node"},
    )
    workflow.add_edge("edit_reviewer_tools", "edit_reviewer_node")
    workflow.add_conditional_edges(
      "edit_reviewer_structured_node",
      IssueAnswerAndFixSuccessRouter(),
      {True: "after_edit_build_branch_node", False: "code_editing_node"},
    )

    workflow.add_conditional_edges(
      "after_edit_build_branch_node",
      IssueAnswerAndFixNeedBuildRouter(),
      {True: "after_edit_build_node", False: "after_edit_test_branch_node"},
    )
    if is_using_user_defined_container:
      workflow.add_edge("after_edit_build_node", "after_edit_general_build_structured_node")
    else:
      workflow.add_conditional_edges(
        "after_edit_build_node",
        functools.partial(tools_condition, messages_key="build_messages"),
        {
          "tools": "after_edit_build_tools",
          END: "after_edit_general_build_structured_node",
        },
      )
      workflow.add_edge("after_edit_build_tools", "after_edit_build_node")
    workflow.add_conditional_edges(
      "after_edit_general_build_structured_node",
      IssueAnswerAndFixSuccessRouter(),
      {True: "after_edit_test_branch_node", False: "code_editing_node"},
    )
    workflow.add_conditional_edges(
      "after_edit_test_branch_node",
      IssueAnswerAndFixNeedTestRouter(),
      {True: "after_edit_test_node", False: "issue_responder_node"},
    )
    if is_using_user_defined_container:
      workflow.add_edge("after_edit_test_node", "after_edit_general_test_structured_node")
    else:
      workflow.add_conditional_edges(
        "after_edit_test_node",
        functools.partial(tools_condition, messages_key="test_messages"),
        {"tools": "after_edit_test_tools", END: "after_edit_general_test_structured_node"},
      )
      workflow.add_edge("after_edit_test_tools", "after_edit_test_node")
    workflow.add_conditional_edges(
      "after_edit_general_test_structured_node",
      IssueAnswerAndFixSuccessRouter(),
      {True: "issue_responder_node", False: "code_editing_node"},
    )

    workflow.add_edge("issue_responder_node", END)

    workflow.set_entry_point("issue_to_query_node")
    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(
    self,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    response_mode: ResponseModeEnum,
    run_build: bool,
    run_test: bool,
    thread_id: Optional[str] = None,
    recursion_limit: int = 200,
  ):
    config = {}
    config["recursion_limit"] = recursion_limit
    if thread_id:
      config["configurable"] = {}
      config["configurable"]["thread_id"] = thread_id

    if response_mode != ResponseModeEnum.ONLY_ANSWER and (run_build or run_test):
      self.container.build_docker_image()
      self.container.start_container()

    output_state = self.subgraph.invoke(
      {
        "issue_title": issue_title,
        "issue_body": issue_body,
        "issue_comments": issue_comments,
        "response_mode": response_mode,
        "run_build": run_build,
        "run_test": run_test,
        "project_path": str(self.local_path),
        "project_structure": self.project_structure,
        "build_messages": [],
        "exist_build": run_build,
        "build_command_summary": "",
        "build_fail_log": "",
        "test_messages": [],
        "exist_test": run_test,
        "test_command_summary": "",
        "test_fail_log": "",
        "code_edit_messages": [],
        "patch": "",
        "issue_response": "",
      },
      config,
    )
    if response_mode != ResponseModeEnum.ONLY_ANSWER and (run_build or run_test):
      self.container.cleanup()
    return output_state["issue_response"], output_state.get("patch", "")
