from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueState
from prometheus.lang_graph.subgraphs.bug_reproduction_subgraph import BugReproductionSubgraph


class BugReproductionSubgraphNode:
  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    kg: KnowledgeGraph,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.bug_reproduction_subgraph = BugReproductionSubgraph(
      model, container, kg, thread_id, checkpointer
    )

  def __call__(self, state: IssueState):
    output_state = self.bug_reproduction_subgraph.invoke(
      state["issue_title"],
      state["issue_body"],
      state["issue_comments"],
      state["bug_summary"],
    )
    return {
      "reproduced_bug": output_state["reproduced_bug"],
      "reproduced_bug_file": output_state["reproduced_bug_file"],
      "reproduced_bug_commands": output_state["reproduced_bug_commands"],
    }
