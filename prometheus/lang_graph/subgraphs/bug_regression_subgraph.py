from typing import Mapping, Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.constants import END
from langgraph.graph import StateGraph

from prometheus.docker.base_container import BaseContainer
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.context_retrieval_subgraph_node import ContextRetrievalSubgraphNode
from prometheus.lang_graph.nodes.git_reset_node import GitResetNode
from prometheus.lang_graph.nodes.issue_bug_regression_check_result_node import (
    IssueBugRegressionCheckResultNode,
)
from prometheus.lang_graph.nodes.issue_bug_regression_context_message_node import (
    IssueBugRegressionContextMessageNode,
)
from prometheus.lang_graph.nodes.issue_bug_regression_patch_update_node import (
    IssueBugRegressionPatchUpdateNode,
)
from prometheus.lang_graph.nodes.issue_bug_regression_tests_selection_node import (
    IssueBugRegressionTestsSelectionNode,
)
from prometheus.lang_graph.nodes.run_regression_tests_subgraph_node import (
    RunRegressionTestsSubgraphNode,
)
from prometheus.lang_graph.nodes.update_container_node import UpdateContainerNode
from prometheus.lang_graph.subgraphs.bug_regression_state import BugRegressionState


class BugRegressionSubgraph:
    """
    This class defines a LangGraph-based state machine that automatically selects and runs regression tests
    for GitHub issues. It orchestrates context retrieval, tests selection, test execution, and feedback loops.
    """

    def __init__(
        self,
        advanced_model: BaseChatModel,
        base_model: BaseChatModel,
        container: BaseContainer,
        kg: KnowledgeGraph,
        git_repo: GitRepository,
        neo4j_driver: neo4j.Driver,
        max_token_per_neo4j_result: int,
    ):
        """
        Initialize the run regression tests pipeline with all necessary parts.

        Args:
            advanced_model: More powerful LLM for structured reasoning and synthesis.
            base_model: Lighter LLM for simpler tasks (e.g., file selection).
            container: Docker-based sandbox for running code.
            kg: Codebase knowledge graph used for context retrieval.
            git_repo: Git repository interface for codebase manipulation.
            neo4j_driver: Neo4j driver used for graph traversal.
            max_token_per_neo4j_result: Truncation budget per retrieved context chunk.
        """

        # Step 1: Generate initial system messages based on issue data
        issue_bug_regression_context_message_node = IssueBugRegressionContextMessageNode()

        # Step 2: Retrieve relevant code/documentation context from the knowledge graph
        context_retrieval_subgraph_node = ContextRetrievalSubgraphNode(
            base_model,
            kg,
            git_repo.playground_path,
            neo4j_driver,
            max_token_per_neo4j_result,
            "select_regression_query",
            "select_regression_context",
        )
        # Step 3: Select relevant regression tests based on the issue and retrieved context
        issue_bug_regression_tests_selection_node = IssueBugRegressionTestsSelectionNode(
            model=advanced_model
        )
        # Step 4: Run the selected regression tests before the fix and store results
        before_run_regression_tests_subgraph_node = RunRegressionTestsSubgraphNode(
            model=base_model,
            container=container,
            passed_regression_tests_key="before_passed_regression_tests",
        )
        # Step 5: Update the Git repository with the first untested patch and update the container
        issue_bug_regression_patch_update_node = IssueBugRegressionPatchUpdateNode(
            git_repo=git_repo
        )
        update_container = UpdateContainerNode(container=container, git_repo=git_repo)
        # Step 6: Run the regression tests after applying the patch and store results
        after_run_regression_tests_subgraph_node = RunRegressionTestsSubgraphNode(
            model=base_model,
            container=container,
            passed_regression_tests_key="after_passed_regression_tests",
        )
        # Step 7: Check if the applied patch passes all regression tests and update the state
        issue_bug_regression_check_result_node = IssueBugRegressionCheckResultNode()

        # Step 8: Reset the Repository to a clean state
        git_reset_node = GitResetNode(git_repo=git_repo)

        # Define the state machine
        workflow = StateGraph(BugRegressionState)

        # Add nodes to the state machine
        workflow.add_node(
            "issue_bug_regression_context_message_node",
            issue_bug_regression_context_message_node,
        )
        workflow.add_node("context_retrieval_subgraph_node", context_retrieval_subgraph_node)
        workflow.add_node(
            "issue_bug_regression_tests_selection_node", issue_bug_regression_tests_selection_node
        )
        workflow.add_node(
            "before_run_regression_tests_subgraph_node", before_run_regression_tests_subgraph_node
        )
        workflow.add_node(
            "issue_bug_regression_patch_update_node", issue_bug_regression_patch_update_node
        )
        workflow.add_node("update_container_node", update_container)
        workflow.add_node(
            "after_run_regression_tests_subgraph_node", after_run_regression_tests_subgraph_node
        )
        workflow.add_node(
            "issue_bug_regression_check_result_node", issue_bug_regression_check_result_node
        )
        workflow.add_node("git_reset_node", git_reset_node)

        # Define transitions between nodes
        workflow.set_entry_point("issue_bug_regression_context_message_node")
        workflow.add_edge(
            "issue_bug_regression_context_message_node", "context_retrieval_subgraph_node"
        )
        workflow.add_edge(
            "context_retrieval_subgraph_node", "issue_bug_regression_tests_selection_node"
        )
        workflow.add_edge(
            "issue_bug_regression_tests_selection_node", "before_run_regression_tests_subgraph_node"
        )
        workflow.add_conditional_edges(
            "before_run_regression_tests_subgraph_node",
            lambda state: state["untested_patches"] > 0,
            {
                True: "issue_bug_regression_patch_update_node",
                False: "git_reset_node",
            },
        )
        workflow.add_edge("git_reset_node", END)
        workflow.add_edge("issue_bug_regression_patch_update_node", "update_container_node")
        workflow.add_edge("update_container_node", "after_run_regression_tests_subgraph_node")
        workflow.add_edge(
            "after_run_regression_tests_subgraph_node", "issue_bug_regression_check_result_node"
        )
        workflow.add_conditional_edges(
            "issue_bug_regression_check_result_node",
            lambda state: len(state["untested_patches"]) > 0,
            {
                True: "issue_bug_regression_patch_update_node",
                False: END,
            },
        )

        # Compile the full LangGraph subgraph
        self.subgraph = workflow.compile()

    def invoke(
        self,
        issue_title: str,
        issue_body: str,
        issue_comments: Sequence[Mapping[str, str]],
        patches: Sequence[str],
        number_of_selected_regression_tests: int = 3,
        recursion_limit: int = 120,
    ):
        """
        Run the bug regression subgraph.

        Args:
            issue_title: Title of the GitHub issue.
            issue_body: Main body text describing the bug.
            issue_comments: List of user/system comments for context.
            patches: List of patches to add to the issue.
            number_of_selected_regression_tests: Number of regression tests to select.
            recursion_limit: Max steps before triggering recovery fallback.
        Returns:
            The result of the bug regression process
        """
        config = {"recursion_limit": recursion_limit}

        input_state = {
            "issue_title": issue_title,
            "issue_body": issue_body,
            "issue_comments": issue_comments,
            "max_refined_query_loop": 3,
            "number_of_selected_regression_tests": number_of_selected_regression_tests,
            "untested_patches": patches,
        }

        output_state = self.subgraph.invoke(input_state, config)
        # return the regression failure log for debugging
        return {
            "passed_patches": output_state["passed_patches"],
            "regression_failure_log": output_state["regression_failure_log"],
        }
