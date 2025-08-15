import logging
import threading
from typing import Dict

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel

from prometheus.docker.base_container import BaseContainer
from prometheus.git.git_repository import GitRepository
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_get_regression_tests_subgraph import (
    BugGetRegressionTestsSubgraph,
)


class BugGetRegressionTestsSubgraphNode:
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
        self._logger = logging.getLogger(
            f"thread-{threading.get_ident()}.prometheus.lang_graph.nodes.bug_get_regression_tests_subgraph_node"
        )
        self.subgraph = BugGetRegressionTestsSubgraph(
            advanced_model=advanced_model,
            base_model=base_model,
            container=container,
            kg=kg,
            git_repo=git_repo,
            neo4j_driver=neo4j_driver,
            max_token_per_neo4j_result=max_token_per_neo4j_result,
        )

    def __call__(self, state: Dict):
        self._logger.info("Enter bug_get_regression_tests_subgraph_node")

        output_state = self.subgraph.invoke(
            issue_title=state["issue_title"],
            issue_body=state["issue_body"],
            issue_comments=state["issue_comments"],
        )
        self._logger.debug(
            f"Selected {len(output_state['regression_tests'])} regression tests: {output_state['regression_tests']}"
        )
        return {"regression_tests": output_state["regression_tests"]}
