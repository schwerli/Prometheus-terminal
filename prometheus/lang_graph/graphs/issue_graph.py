from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from prometheus.docker.base_container import BaseContainer
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueState, IssueType
from prometheus.lang_graph.nodes.issue_bug_subgraph_node import IssueBugSubgraphNode
from prometheus.lang_graph.nodes.issue_classification_subgraph_node import (
    IssueClassificationSubgraphNode,
)
from prometheus.lang_graph.nodes.noop_node import NoopNode


class IssueGraph:
    def __init__(
        self,
        advanced_model: BaseChatModel,
        base_model: BaseChatModel,
        kg: KnowledgeGraph,
        git_repo: GitRepository,
        neo4j_driver: neo4j.Driver,
        max_token_per_neo4j_result: int,
        container: BaseContainer,
        build_commands: Optional[Sequence[str]] = None,
        test_commands: Optional[Sequence[str]] = None,
    ):
        self.git_repo = git_repo

        issue_type_branch_node = NoopNode()
        issue_classification_subgraph_node = IssueClassificationSubgraphNode(
            model=base_model,
            kg=kg,
            neo4j_driver=neo4j_driver,
            max_token_per_neo4j_result=max_token_per_neo4j_result,
        )
        issue_bug_subgraph_node = IssueBugSubgraphNode(
            advanced_model=advanced_model,
            base_model=base_model,
            container=container,
            kg=kg,
            git_repo=git_repo,
            neo4j_driver=neo4j_driver,
            max_token_per_neo4j_result=max_token_per_neo4j_result,
            build_commands=build_commands,
            test_commands=test_commands,
        )

        workflow = StateGraph(IssueState)

        workflow.add_node("issue_type_branch_node", issue_type_branch_node)
        workflow.add_node("issue_classification_subgraph_node", issue_classification_subgraph_node)
        workflow.add_node("issue_bug_subgraph_node", issue_bug_subgraph_node)

        workflow.set_entry_point("issue_type_branch_node")
        workflow.add_conditional_edges(
            "issue_type_branch_node",
            lambda state: state["issue_type"],
            {
                IssueType.AUTO: "issue_classification_subgraph_node",
                IssueType.BUG: "issue_bug_subgraph_node",
                IssueType.FEATURE: END,
                IssueType.DOCUMENTATION: END,
                IssueType.QUESTION: END,
            },
        )
        workflow.add_conditional_edges(
            "issue_classification_subgraph_node",
            lambda state: state["issue_type"],
            {
                IssueType.BUG: "issue_bug_subgraph_node",
                IssueType.FEATURE: END,
                IssueType.DOCUMENTATION: END,
                IssueType.QUESTION: END,
            },
        )
        workflow.add_edge("issue_bug_subgraph_node", END)

        self.graph = workflow.compile()

    def invoke(
        self,
        issue_title: str,
        issue_body: str,
        issue_comments: Sequence[Mapping[str, str]],
        issue_type: IssueType,
        run_build: bool,
        run_existing_test: bool,
        number_of_candidate_patch: int,
    ):
        config = None

        input_state = {
            "issue_title": issue_title,
            "issue_body": issue_body,
            "issue_comments": issue_comments,
            "issue_type": issue_type,
            "run_build": run_build,
            "run_existing_test": run_existing_test,
            "number_of_candidate_patch": number_of_candidate_patch,
        }

        output_state = self.graph.invoke(input_state, config)

        self.git_repo.reset_repository()

        return output_state
