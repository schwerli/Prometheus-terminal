from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from prometheus.docker.base_container import BaseContainer
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.bug_reproduction_subgraph_node import BugReproductionSubgraphNode
from prometheus.lang_graph.nodes.issue_bug_responder_node import IssueBugResponderNode
from prometheus.lang_graph.nodes.issue_not_verified_bug_subgraph_node import (
  IssueNotVerifiedBugSubgraphNode,
)
from prometheus.lang_graph.nodes.issue_verified_bug_subgraph_node import (
  IssueVerifiedBugSubgraphNode,
)
from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState


class IssueBugSubgraph:
  def __init__(
    self,
    model: BaseChatModel,
    container: BaseContainer,
    kg: KnowledgeGraph,
    git_repo: GitRepository,
    neo4j_driver: neo4j.Driver,
    max_token_per_neo4j_result: int,
    build_commands: Optional[Sequence[str]] = None,
    test_commands: Optional[Sequence[str]] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
  ):
    self.thread_id = thread_id

    bug_reproduction_subgraph_node = BugReproductionSubgraphNode(
      model,
      container,
      kg,
      git_repo,
      neo4j_driver,
      max_token_per_neo4j_result,
      test_commands,
      thread_id,
      checkpointer,
    )

    issue_verified_bug_subgraph_node = IssueVerifiedBugSubgraphNode(
      model=model,
      container=container,
      kg=kg,
      git_repo=git_repo,
      neo4j_driver=neo4j_driver,
      max_token_per_neo4j_result=max_token_per_neo4j_result,
      build_commands=build_commands,
      test_commands=test_commands,
      thread_id=thread_id,
      checkpointer=checkpointer,
    )
    issue_not_verified_bug_subgraph_node = IssueNotVerifiedBugSubgraphNode(
      model=model,
      kg=kg,
      git_repo=git_repo,
      neo4j_driver=neo4j_driver,
      max_token_per_neo4j_result=max_token_per_neo4j_result,
      thread_id=thread_id,
      checkpointer=checkpointer,
    )

    issue_bug_responder_node = IssueBugResponderNode(model)

    workflow = StateGraph(IssueBugState)

    workflow.add_node("bug_reproduction_subgraph_node", bug_reproduction_subgraph_node)

    workflow.add_node("issue_verified_bug_subgraph_node", issue_verified_bug_subgraph_node)
    workflow.add_node("issue_not_verified_bug_subgraph_node", issue_not_verified_bug_subgraph_node)

    workflow.add_node("issue_bug_responder_node", issue_bug_responder_node)

    workflow.set_entry_point("bug_reproduction_subgraph_node")

    workflow.add_conditional_edges(
      "bug_reproduction_subgraph_node",
      lambda state: state["reproduced_bug"] or state["run_build"] or state["run_existing_test"],
      {True: "issue_verified_bug_subgraph_node", False: "issue_not_verified_bug_subgraph_node"},
    )
    workflow.add_conditional_edges(
      "issue_verified_bug_subgraph_node",
      lambda state: bool(state["edit_patch"]),
      {True: "issue_bug_responder_node", False: "issue_not_verified_bug_subgraph_node"},
    )
    workflow.add_edge("issue_not_verified_bug_subgraph_node", "issue_bug_responder_node")
    workflow.add_edge("issue_bug_responder_node", END)

    self.subgraph = workflow.compile(checkpointer=checkpointer)

  def invoke(
    self,
    issue_title: str,
    issue_body: str,
    issue_comments: Sequence[Mapping[str, str]],
    run_build: bool,
    run_existing_test: bool,
    number_of_candidate_patch: int,
    recursion_limit: int = 30,
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
      "number_of_candidate_patch": number_of_candidate_patch,
    }

    output_state = self.subgraph.invoke(input_state, config)
    return {
      "edit_patch": output_state["edit_patch"],
      "passed_reproducing_test": output_state["passed_reproducing_test"],
      "passed_build": output_state["passed_build"],
      "passed_existing_test": output_state["passed_existing_test"],
      "issue_response": output_state["issue_response"],
    }
