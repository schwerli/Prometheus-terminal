import functools
from typing import Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.docker.base_container import BaseContainer
from prometheus.lang_graph.nodes.bug_fix_verify_node import BugFixVerifyNode
from prometheus.lang_graph.nodes.bug_fix_verify_structured_node import BugFixVerifyStructuredNode
from prometheus.lang_graph.subgraphs.bug_fix_verification_state import BugFixVerficationState


class BugFixVerificationSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.thread_id = thread_id

    bug_fix_verify_node = BugFixVerifyNode(model, container)
    bug_fix_verify_tools = ToolNode(
      tools=bug_fix_verify_node.tools,
      name="bug_fix_verify_tools",
      messages_key="bug_fix_verify_messages",
    )
    bug_fix_verify_structured_node = BugFixVerifyStructuredNode(model)

    workflow = StateGraph(BugFixVerficationState)

    workflow.add_node("bug_fix_verify_node", bug_fix_verify_node)
    workflow.add_node("bug_fix_verify_tools", bug_fix_verify_tools)
    workflow.add_node("bug_fix_verify_structured_node", bug_fix_verify_structured_node)

    workflow.set_entry_point("bug_fix_verify_node")

    workflow.add_conditional_edges(
      "bug_fix_verify_node",
      functools.partial(tools_condition, messages_key="bug_fix_verify_messages"),
      {
        "tools": "bug_fix_verify_tools",
        END: "bug_fix_verify_structured_node",
      },
    )
    workflow.add_edge("bug_fix_verify_tools", "bug_fix_verify_node")
    workflow.add_edge("bug_fix_verify_structured_node", END)

    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(self, reproduced_bug_file: str, reproduced_bug_commands: Sequence[str]):
    config = None
    if self.thread_id:
      config = {"configurable": {"thread_id": self.thread_id}}

    input_state = {
      "reproduced_bug_file": reproduced_bug_file,
      "reproduced_bug_commands": reproduced_bug_commands,
    }

    output_state = self.subgraph.invoke(input_state, config)

    return {
      "reproducing_test_passed": output_state["reproducing_test_passed"],
      "reproducing_test_fail_log": output_state["reproducing_test_fail_log"],
    }