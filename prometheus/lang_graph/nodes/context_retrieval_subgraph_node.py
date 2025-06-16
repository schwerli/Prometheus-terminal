import logging
from typing import Dict

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.context_retrieval_subgraph import ContextRetrievalSubgraph


class ContextRetrievalSubgraphNode:
    def __init__(
        self,
        model: BaseChatModel,
        kg: KnowledgeGraph,
        neo4j_driver: neo4j.Driver,
        max_token_per_neo4j_result: int,
        query_key_name: str,
        context_key_name: str,
    ):
        self._logger = logging.getLogger(
            "prometheus.lang_graph.nodes.context_retrieval_subgraph_node"
        )
        self.context_retrieval_subgraph = ContextRetrievalSubgraph(
            model=model,
            kg=kg,
            neo4j_driver=neo4j_driver,
            max_token_per_neo4j_result=max_token_per_neo4j_result,
        )
        self.query_key_name = query_key_name
        self.context_key_name = context_key_name

    def __call__(self, state: Dict, max_tries: int = 3) -> Dict[str, str]:
        self._logger.info("Enter context retrieval subgraph")
        output_state = None
        # Trying several times to retrieve context
        for attempt in range(1, max_tries + 1):
            try:
                output_state = self.context_retrieval_subgraph.invoke(
                    state[self.query_key_name], state["max_refined_query_loop"]
                )
                break
            except Exception as e:
                if attempt < max_tries:
                    self._logger.warning(
                        f"Context retrieval failed, retrying {attempt}/{max_tries} times: {e}"
                    )
        if output_state is None:
            self._logger.error("Context retrieval failed after maximum attempts")
            raise RuntimeError("Failed to retrieve context after maximum attempts")
        self._logger.info(f"Context retrieved: {output_state['context']}")
        return {self.context_key_name: output_state["context"]}
