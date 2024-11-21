import functools
from typing import Mapping, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.bug_reproducing_execute_node import BugReproducingExecuteNode
from prometheus.lang_graph.nodes.bug_reproducing_structured_node import BugReproducingStructuredNode
from prometheus.lang_graph.nodes.bug_reproducing_write_node import BugReproducingWriteNode
from prometheus.lang_graph.nodes.reset_messages_node import ResetMessagesNode
from prometheus.lang_graph.nodes.update_container_node import UpdateContainerNode
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState


class BugReproductionSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    kg: KnowledgeGraph,
    test_commands: Optional[Sequence[str]] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.thread_id = thread_id

    bug_reproducing_write_node = BugReproducingWriteNode(model, kg)
    bug_reproducing_write_tools = ToolNode(
      tools=bug_reproducing_write_node.tools,
      name="bug_reproducing_write_tools",
      messages_key="bug_reproducing_write_messages",
    )
    update_container_node = UpdateContainerNode(container, kg)
    bug_reproducing_execute_node = BugReproducingExecuteNode(model, container, kg, test_commands)
    bug_reproducing_execute_tools = ToolNode(
      tools=bug_reproducing_execute_node.tools,
      name="bug_reproducing_execute_tools",
      messages_key="bug_reproducing_execute_messages",
    )
    bug_reproducing_structured_node = BugReproducingStructuredNode(model)
    reset_bug_reproducing_write_messages_node = ResetMessagesNode("bug_reproducing_write_messages")
    reset_bug_reproducing_execute_messages_node = ResetMessagesNode(
      "bug_reproducing_execute_messages"
    )

    workflow = StateGraph(BugReproductionState)

    workflow.add_node("bug_reproducing_write_node", bug_reproducing_write_node)
    workflow.add_node("bug_reproducing_write_tools", bug_reproducing_write_tools)
    workflow.add_node("update_container_node", update_container_node)
    workflow.add_node("bug_reproducing_execute_node", bug_reproducing_execute_node)
    workflow.add_node("bug_reproducing_execute_tools", bug_reproducing_execute_tools)
    workflow.add_node("bug_reproducing_structured_node", bug_reproducing_structured_node)
    workflow.add_node(
      "reset_bug_reproducing_write_messages_node", reset_bug_reproducing_write_messages_node
    )
    workflow.add_node(
      "reset_bug_reproducing_execute_messages_node", reset_bug_reproducing_execute_messages_node
    )

    workflow.set_entry_point("bug_reproducing_write_node")

    workflow.add_conditional_edges(
      "bug_reproducing_write_node",
      functools.partial(tools_condition, messages_key="bug_reproducing_write_messages"),
      {
        "tools": "bug_reproducing_write_tools",
        END: "update_container_node",
      },
    )
    workflow.add_edge("bug_reproducing_write_tools", "bug_reproducing_write_node")
    workflow.add_edge("update_container_node", "bug_reproducing_execute_node")
    workflow.add_conditional_edges(
      "bug_reproducing_execute_node",
      functools.partial(tools_condition, messages_key="bug_reproducing_execute_messages"),
      {
        "tools": "bug_reproducing_execute_tools",
        END: "bug_reproducing_structured_node",
      },
    )
    workflow.add_edge("bug_reproducing_execute_tools", "bug_reproducing_execute_node")
    workflow.add_conditional_edges(
      "bug_reproducing_structured_node",
      lambda state: state["reproduced_bug"],
      {True: END, False: "reset_bug_reproducing_write_messages_node"},
    )
    workflow.add_edge(
      "reset_bug_reproducing_write_messages_node", "reset_bug_reproducing_execute_messages_node"
    )
    workflow.add_edge("reset_bug_reproducing_execute_messages_node", "bug_reproducing_write_node")

    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(
    self,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    bug_context: str,
    recursion_limit: int = 50,
  ):
    config = {"recursion_limit": recursion_limit}
    if self.thread_id:
      config["configurable"] = {"thread_id": self.thread_id}

    input_state = {
      "issue_title": issue_title,
      "issue_body": issue_body,
      "issue_comments": issue_comments,
      "bug_context": bug_context,
    }

    output_state = self.subgraph.invoke(input_state, config)

    return {
      "reproduced_bug": output_state["reproduced_bug"],
      "reproduced_bug_file": output_state["reproduced_bug_file"],
      "reproduced_bug_commands": output_state["reproduced_bug_commands"],
    }
