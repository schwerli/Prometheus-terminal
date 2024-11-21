from typing import Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.build_and_test_subgraph import BuildAndTestSubgraph
from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState


class BuildAndTestSubgraphNode:
  def __init__(
    self,
    container: BaseContainer,
    model: BaseChatModel,
    kg: KnowledgeGraph,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.build_and_test_subgraph = BuildAndTestSubgraph(
      container, model, kg, build_commands, test_commands, thread_id, checkpointer
    )

  def __call__(self, state: IssueBugState):
    exist_build = None
    build_command_summary = None
    build_fail_log = None
    exist_test = None
    test_command_summary = None
    existing_test_fail_log = None

    if "build_command_summary" in state and state["build_command_summary"]:
      exist_build = state["exist_build"]
      build_command_summary = state["build_command_summary"]
      build_fail_log = state["build_fail_log"]

    if "test_command_summary" in state and state["test_command_summary"]:
      exist_test = state["exist_test"]
      test_command_summary = state["test_command_summary"]
      existing_test_fail_log = state["existing_test_fail_log"]

    output_state = self.build_and_test_subgraph.invoke(
      state["run_build"],
      state["run_existing_test"],
      exist_build,
      build_command_summary,
      build_fail_log,
      exist_test,
      test_command_summary,
      existing_test_fail_log,
    )

    return {
      "exist_build": output_state["exist_build"],
      "build_command_summary": output_state["build_command_summary"],
      "build_fail_log": output_state["build_fail_log"],
      "exist_test": output_state["exist_test"],
      "test_command_summary": output_state["test_command_summary"],
      "existing_test_fail_log": output_state["test_fail_log"],
    }
