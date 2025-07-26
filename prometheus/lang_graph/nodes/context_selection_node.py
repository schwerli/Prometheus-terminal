import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.context_retrieval_state import ContextRetrievalState
from prometheus.utils.lang_graph_util import extract_last_tool_messages
from prometheus.utils.neo4j_util import neo4j_data_for_context_generator


class ContextSelectionStructuredOutput(BaseModel):
    reasoning: str = Field(
        description="Your step-by-step reasoning why the context is relevant to the query"
    )
    relevant: bool = Field(description="If the context is relevant to the query")


class ContextSelectionNode:
    SYS_PROMPT = """\
You are a context selection agent that evaluates if a piece of code context is relevant to a given query. Your goal is to determine if the context directly answers the query requirements.

Your evaluation must consider two key aspects:
1. Query Match: Does the context directly address specific requirements mentioned in the query?
2. Extended relevance: Does this context provide essential information needed to understand the query topic?

Follow these strict evaluation steps:
1. First, identify specific requirements in the query
2. Check if the context directly addresses these requirements
3. Consider if it provides essential context by examining:
   - Function dependencies
   - Type definitions
   - Configuration requirements
   - Implementation details needed for completeness

Query relevance guidelines - include only if:
- It directly implements functionality mentioned in the query
- It contains specific elements the query asks about
- It's necessary to understand or implement query requirements
- It provides critical information needed to answer the query

Remember: Your primary goal is to determine if this specific piece of context directly helps answer the query requirements.

Provide your analysis in a structured format matching the ContextSelectionStructuredOutput model.

Example:

Query: "How does the login endpoint validate passwords?"

Context to evaluate:
```python
def validate_password(password: str, hash: str):
    return bcrypt.checkpw(password.encode(), hash.encode())
```

Example output:
```json
{
  "reasoning": "1. Query requirement analysis:
   - Query specifically asks about password validation
   - Needs implementation details of validation process
   2. Context evaluation:
   - Directly implements the password validation function
   - Shows exactly how passwords are compared using bcrypt
   - Provides essential implementation detail
   3. Relevance confirmation:
   - Directly answers how password validation works
   - Shows specific security mechanism (bcrypt) used",
  "relevant": true
}
```

Your task is to analyze the context and provide a similar structured output with detailed reasoning and a relevance decision.
""".replace("{", "{{").replace("}", "}}")

    HUMAN_PROMPT = """\
Query:
{query}

Found context:
{context}

Please classify if the found context is relevant to the query.
"""

    def __init__(self, model: BaseChatModel):
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.SYS_PROMPT), ("human", "{human_prompt}")]
        )
        structured_llm = model.with_structured_output(ContextSelectionStructuredOutput)
        self.model = prompt | structured_llm
        self._logger = logging.getLogger("prometheus.lang_graph.nodes.context_selection_node")

    def format_human_prompt(self, state: ContextRetrievalState, search_result: str) -> str:
        context_info = self.HUMAN_PROMPT.format(query=state["query"], context=search_result)
        return context_info

    def __call__(self, state: ContextRetrievalState):
        self._logger.info("Starting context selection process")
        context_list = state.get("context", [])
        for tool_message in extract_last_tool_messages(state["context_provider_messages"]):
            for context in neo4j_data_for_context_generator(tool_message.artifact):
                context_str = str(context)
                human_prompt = self.format_human_prompt(state, context_str)
                response = self.model.invoke({"human_prompt": human_prompt})
                self._logger.debug(
                    f"Is this search result {context_str} relevant?: {response.relevant}"
                )
                if response.relevant:
                    context_list.append(context)
        self._logger.info(f"Context selection complete, returning context {context_list}")
        return {"context": context_list}
