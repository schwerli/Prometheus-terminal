from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueType
from prometheus.lang_graph.nodes.bug_reproduction_subgraph_node import BugReproductionSubgraphNode
from prometheus.lang_graph.nodes.issue_to_context_node import IssueToContextNode
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
      model, container, kg, thread_id, checkpointer
    )

    workflow = StateGraph(IssueBugState)

    workflow.add_node("issue_to_bug_context_node", issue_to_bug_context_node)
    workflow.add_node("bug_reproduction_subgraph_node", bug_reproduction_subgraph_node)

    workflow.set_entry_point("issue_to_bug_context_node")
    workflow.add_edge("issue_to_bug_context_node", "bug_reproduction_subgraph_node")
    workflow.add_edge("bug_reproduction_subgraph_node", END)

    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(self, issue_title: str, issue_body: str, issue_comments: Sequence[Mapping[str, str]]):
    config = None
    if self.thread_id:
      config = {"configurable": {"thread_id": self.thread_id}}

    if not self.container.is_running():
      self.container.build_docker_image()
      self.container.start_container()

    input_state = {
      "issue_title": issue_title,
      "issue_body": issue_body,
      "issue_comments": issue_comments,
    }

    output_state = self.subgraph.invoke(input_state, config)

    self.container.cleanup()

    return {
      "bug_context": output_state["bug_context"],
      "reproduced_bug": output_state["reproduced_bug"],
      "reproduced_bug_file": output_state["reproduced_bug_file"],
      "reproduced_bug_commands": output_state["reproduced_bug_commands"],
    }
