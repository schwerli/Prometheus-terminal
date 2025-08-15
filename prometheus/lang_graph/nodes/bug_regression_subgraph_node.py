import logging
import threading
from typing import Dict

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel

from prometheus.docker.base_container import BaseContainer
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_regression_subgraph import BugRegressionSubgraph


class BugRegressionSubgraphNode:
    def __init__(
        self,
        advanced_model: BaseChatModel,
        base_model: BaseChatModel,
        container: BaseContainer,
        kg: KnowledgeGraph,
        git_repo: GitRepository,
        neo4j_driver: neo4j.Driver,
        max_token_per_neo4j_result: int,
        testing_patches_key: str,
        is_testing_patch_list: bool = True,
    ):
        self._logger = logging.getLogger(
            f"thread-{threading.get_ident()}.prometheus.lang_graph.nodes.bug_regression_subgraph_node"
        )
        self.subgraph = BugRegressionSubgraph(
            advanced_model=advanced_model,
            base_model=base_model,
            container=container,
            kg=kg,
            git_repo=git_repo,
            neo4j_driver=neo4j_driver,
            max_token_per_neo4j_result=max_token_per_neo4j_result,
        )
        self.testing_patches_key = testing_patches_key
        self.is_testing_patch_list = is_testing_patch_list

    def __call__(self, state: Dict):
        self._logger.info("Enter bug_regression_subgraph_node")
        self._logger.debug(f"Testing: {state[self.testing_patches_key]}")

        output_state = self.subgraph.invoke(
            issue_title=state["issue_title"],
            issue_body=state["issue_body"],
            issue_comments=state["issue_comments"],
            patches=state[self.testing_patches_key]
            if self.is_testing_patch_list
            else [state[self.testing_patches_key]],
        )

        self._logger.info(f"passed_patches: {output_state['passed_patches']}")
        if not self.is_testing_patch_list:
            self._logger.debug(
                f"regression_test_fail_log: {output_state['regression_test_fail_log']}"
            )
            return {
                "passed_regression_tests_patches": True
                if output_state["passed_patches"]
                else False,
                "regression_test_fail_log": output_state["regression_test_fail_log"],
            }
        else:
            return {
                "passed_regression_tests_patches": output_state["passed_patches"],
            }
