"""Normalized Not Verified Bug Subgraph

This module implements a simplified enhanced issue not verified bug subgraph
with patch normalization and deduplication, using standard final patch selection.
"""

import logging
import threading
from typing import Mapping, Optional, Sequence

import neo4j
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from prometheus.container.base_container import BaseContainer
from prometheus.knowledge_graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.graphs.issue_state import IssueNotVerifiedBugState
from prometheus.lang_graph.nodes.context_retrieval_subgraph_node import ContextRetrievalSubgraphNode
from prometheus.lang_graph.nodes.edit_message_node import EditMessageNode
from prometheus.lang_graph.nodes.edit_node import EditNode
from prometheus.lang_graph.nodes.final_patch_selection_node import FinalPatchSelectionNode
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode
from prometheus.lang_graph.nodes.git_reset_node import GitResetNode
from prometheus.lang_graph.nodes.issue_bug_analyzer_message_node import IssueBugAnalyzerMessageNode
from prometheus.lang_graph.nodes.issue_bug_analyzer_node import IssueBugAnalyzerNode
from prometheus.lang_graph.nodes.issue_bug_context_message_node import IssueBugContextMessageNode
from prometheus.lang_graph.nodes.patch_normalization_node import PatchNormalizationNode
from prometheus.lang_graph.nodes.reset_messages_node import ResetMessagesNode
from prometheus.repository.git_repository import GitRepository


class NormalizedNotVerifiedBugSubgraph:
    """Simplified Enhanced Issue Not Verified Bug Subgraph

    Simplified workflow with patch normalization and deduplication:
    1. Original context retrieval and bug analysis
    2. Patch generation and diff
    3. Patch normalization and deduplication
    4. Standard final patch selection
    """

    def __init__(
        self,
        advanced_model: BaseChatModel,
        base_model: BaseChatModel,
        kg: KnowledgeGraph,
        git_repo: GitRepository,
        neo4j_driver: neo4j.Driver,
        max_token_per_neo4j_result: int,
        container: Optional[BaseContainer] = None,
    ):
        self._logger = logging.getLogger(
            f"thread-{threading.get_ident()}.prometheus.lang_graph.subgraphs.normalized_not_verified_bug_subgraph"
        )

        # === Initialize Nodes ===
        # Context retrieval subgraph node
        context_retrieval_subgraph_node = ContextRetrievalSubgraphNode(
            advanced_model=advanced_model,
            base_model=base_model,
            kg=kg,
            git_repo=git_repo,
            neo4j_driver=neo4j_driver,
            max_token_per_neo4j_result=max_token_per_neo4j_result,
            container=container,
        )

        # Issue bug context message node
        issue_bug_context_message_node = IssueBugContextMessageNode(
            advanced_model=advanced_model,
            base_model=base_model,
        )

        # Issue bug analyzer message node
        issue_bug_analyzer_message_node = IssueBugAnalyzerMessageNode(
            advanced_model=advanced_model,
            base_model=base_model,
        )

        # Issue bug analyzer node
        issue_bug_analyzer_node = IssueBugAnalyzerNode(
            advanced_model=advanced_model,
            base_model=base_model,
        )

        # Edit message node
        edit_message_node = EditMessageNode(
            advanced_model=advanced_model,
            base_model=base_model,
        )

        # Edit node
        edit_node = EditNode(
            advanced_model=advanced_model,
            base_model=base_model,
        )

        # Git diff node
        git_diff_node = GitDiffNode(
            git_repo=git_repo,
        )

        # Git reset node
        git_reset_node = GitResetNode(
            git_repo=git_repo,
        )

        # Reset messages nodes
        reset_issue_bug_analyzer_messages_node = ResetMessagesNode(
            message_key="issue_bug_analyzer_messages"
        )
        reset_edit_messages_node = ResetMessagesNode(message_key="edit_messages")

        # Patch normalization node (only deduplication)
        patch_normalization_node = PatchNormalizationNode()

        # Final patch selection node (intelligent selection)
        final_patch_selection_node = FinalPatchSelectionNode(model=advanced_model, max_retries=2)

        # === Build Workflow Graph ===
        workflow = StateGraph(IssueNotVerifiedBugState)

        # Add nodes
        workflow.add_node("context_retrieval_subgraph_node", context_retrieval_subgraph_node)
        workflow.add_node("issue_bug_context_message_node", issue_bug_context_message_node)
        workflow.add_node("issue_bug_analyzer_message_node", issue_bug_analyzer_message_node)
        workflow.add_node("issue_bug_analyzer_node", issue_bug_analyzer_node)
        workflow.add_node("edit_message_node", edit_message_node)
        workflow.add_node("edit_node", edit_node)
        workflow.add_node("git_diff_node", git_diff_node)
        workflow.add_node("git_reset_node", git_reset_node)
        workflow.add_node(
            "reset_issue_bug_analyzer_messages_node", reset_issue_bug_analyzer_messages_node
        )
        workflow.add_node("reset_edit_messages_node", reset_edit_messages_node)
        workflow.add_node("patch_normalization_node", patch_normalization_node)
        workflow.add_node("final_patch_selection_node", final_patch_selection_node)

        # === Build Workflow Edges ===
        # Start with context retrieval
        workflow.add_edge("context_retrieval_subgraph_node", "issue_bug_context_message_node")
        workflow.add_edge("issue_bug_context_message_node", "issue_bug_analyzer_message_node")
        workflow.add_edge("issue_bug_analyzer_message_node", "issue_bug_analyzer_node")
        workflow.add_edge("issue_bug_analyzer_node", "edit_message_node")
        workflow.add_edge("edit_message_node", "edit_node")
        workflow.add_edge("edit_node", "git_diff_node")

        # === Decision Point: Continue Generation or Process Patches ===
        workflow.add_conditional_edges(
            "git_diff_node",
            self._routing_logic,
            {
                "continue_generation": "git_reset_node",  # Continue generating more patches
                "process_patches": "patch_normalization_node",  # Process patches with normalization
            },
        )

        # Continue generating patches - original flow
        workflow.add_edge("git_reset_node", "reset_issue_bug_analyzer_messages_node")
        workflow.add_edge("reset_issue_bug_analyzer_messages_node", "reset_edit_messages_node")
        workflow.add_edge("reset_edit_messages_node", "issue_bug_analyzer_message_node")

        # === Patch Processing Flow ===
        # Flow: normalization -> final selection -> END
        workflow.add_edge("patch_normalization_node", "final_patch_selection_node")
        workflow.add_edge("final_patch_selection_node", END)

        self.subgraph = workflow.compile()

    def _routing_logic(self, state: IssueNotVerifiedBugState) -> str:
        """Routing logic to decide whether to continue generation or process patches"""
        patches = state.get("edit_patches", [])
        target_patch_count = state.get("number_of_candidate_patch", 1)
        current_patch_count = len(patches)

        if current_patch_count < target_patch_count:
            return "continue_generation"

        return "process_patches"

    def invoke(
        self,
        issue_title: str,
        issue_body: str,
        issue_comments: Sequence[Mapping[str, str]],
        number_of_candidate_patch: int,
        recursion_limit: int = 500,
    ):
        """Invoke the subgraph with issue information"""
        # Prepare initial state
        initial_state = {
            "issue_title": issue_title,
            "issue_body": issue_body,
            "issue_comments": issue_comments,
            "number_of_candidate_patch": number_of_candidate_patch,
            "edit_patches": [],
            "issue_bug_analyzer_messages": [],
            "edit_messages": [],
        }

        # Execute the workflow
        output_state = self.subgraph.invoke(
            initial_state, config={"recursion_limit": recursion_limit}
        )

        # Extract results
        result = {
            "final_patch": output_state.get("final_patch", ""),
        }

        # Add patch statistics if available
        if "unique_patch_count" in output_state:
            result["patch_statistics"] = {
                "original_patch_count": output_state.get("original_patch_count", 0),
                "unique_patch_count": output_state.get("unique_patch_count", 0),
                "deduplication_ratio": output_state.get("unique_patch_count", 0)
                / max(output_state.get("original_patch_count", 1), 1),
            }

        return result
