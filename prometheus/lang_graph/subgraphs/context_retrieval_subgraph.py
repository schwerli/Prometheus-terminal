import functools
from typing import Sequence

import neo4j
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.context_provider_node import ContextProviderNode
from prometheus.lang_graph.nodes.context_query_message_node import ContextQueryMessageNode
from prometheus.lang_graph.nodes.context_refine_node import ContextRefineNode
from prometheus.lang_graph.nodes.context_selection_node import ContextSelectionNode
from prometheus.lang_graph.nodes.reset_messages_node import ResetMessagesNode
from prometheus.lang_graph.subgraphs.context_retrieval_state import ContextRetrievalState


class ContextRetrievalSubgraph:
    def __init__(
            self,
            model: BaseChatModel,
            kg: KnowledgeGraph,
            neo4j_driver: neo4j.Driver,
            max_token_per_neo4j_result: int,
    ):
        context_query_message_node = ContextQueryMessageNode()
        context_provider_node = ContextProviderNode(
            model, kg, neo4j_driver, max_token_per_neo4j_result
        )
        context_provider_tools = ToolNode(
            tools=context_provider_node.tools,
            name="context_provider_tools",
            messages_key="context_provider_messages",
        )
        context_selection_node = ContextSelectionNode(model)
        reset_context_provider_messages_node = ResetMessagesNode("context_provider_messages")
        context_refine_node = ContextRefineNode(model, kg)

        workflow = StateGraph(ContextRetrievalState)

        workflow.add_node("context_query_message_node", context_query_message_node)
        workflow.add_node("context_provider_node", context_provider_node)
        workflow.add_node("context_provider_tools", context_provider_tools)
        workflow.add_node("context_selection_node", context_selection_node)
        workflow.add_node(
            "reset_context_provider_messages_node", reset_context_provider_messages_node
        )
        workflow.add_node("context_refine_node", context_refine_node)

        workflow.set_entry_point("context_query_message_node")
        workflow.add_edge("context_query_message_node", "context_provider_node")
        workflow.add_conditional_edges(
            "context_provider_node",
            functools.partial(tools_condition, messages_key="context_provider_messages"),
            {"tools": "context_provider_tools", END: "context_selection_node"},
        )
        workflow.add_edge("context_provider_tools", "context_provider_node")
        workflow.add_edge("context_selection_node", "reset_context_provider_messages_node")
        workflow.add_edge("reset_context_provider_messages_node", "context_refine_node")
        workflow.add_conditional_edges(
            "context_refine_node",
            lambda state: bool(state["refined_query"]),
            {True: "context_provider_node", False: END},
        )

        self.subgraph = workflow.compile()

    def invoke(
            self, query: str, max_refined_query_loop: int, recursion_limit: int = 999
    ) -> Sequence[str]:
        config = {"recursion_limit": recursion_limit}

        input_state = {"query": query, "max_refined_query_loop": max_refined_query_loop}

        output_state = self.subgraph.invoke(input_state, config)
        return {"context": output_state["context"]}
