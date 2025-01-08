import logging
from typing import Sequence

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
You are a context selection agent that evaluates if a new piece of code context is relevant to a given query, taking into account previously found relevant context. Your goal is to build a focused, non-redundant set of context that directly answers the query requirements.

Your evaluation must consider three key aspects:
1. Query Match: Does the context directly address specific requirements mentioned in the query?
2. Non-redundancy: Is this information already covered in previously found context?
3. Extended relevance: Does this context provide essential missing information needed to understand previously found relevant context?

Follow these strict evaluation steps:
1. First, identify specific requirements in the query
2. Check if the new context directly addresses these requirements
3. Compare against previously found context to ensure no redundancy:
   - Check for duplicate functionality
   - Look for overlapping implementations
   - Identify similar code patterns
4. Only then consider if it provides essential missing context by examining:
   - Function dependencies
   - Type definitions
   - Configuration requirements
   - Implementation details needed for completeness

Redundancy guidelines - exclude context if:
- It implements the same functionality as existing context
- It contains a subset of information already present
- It only adds non-essential details to existing context

Query relevance guidelines - include only if:
- It directly implements functionality mentioned in the query
- It contains specific elements the query asks about
- It's necessary to understand or implement query requirements
- It provides critical missing information needed by existing relevant context

Remember: Your primary goal is to build the minimal set of context needed to completely answer the query requirements while strictly avoiding redundancy.

Provide your analysis in a structured format matching the ContextSelectionStructuredOutput model.

Example:

Query: "How does the login endpoint validate passwords?"

Previous relevant context:
```python
def login_endpoint(username: str, password: str):
    user = get_user(username)
    if user and validate_password(password, user.password_hash):
        return generate_token(user)
    raise InvalidCredentials()
```

New context to evaluate:
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
   2. New context evaluation:
   - Directly implements the validate_password function used in login_endpoint
   - Shows exactly how passwords are compared using bcrypt
   - Provides essential missing implementation detail
   3. Redundancy check:
   - Previous context only shows the function call
   - New context provides the actual implementation
   - No overlap in functionality
   4. Relevance confirmation:
   - Directly answers how password validation works
   - Completes the understanding of the login process
   - Shows specific security mechanism (bcrypt) used",
  "relevant": true
}
```

Your task is to analyze the new context and provide a similar structured output with detailed reasoning and a relevance decision.
""".replace("{", "{{").replace("}", "}}")

  HUMAN_PROMPT = """\
Query:
{query}

Previous relevant context:
{old_context}

Newly found context:
{new_context}

Please classify if the newly found context is relevant to the query, or extends knowledge found in previously relevant context.
"""

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{human_prompt}")]
    )
    structured_llm = model.with_structured_output(ContextSelectionStructuredOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.context_selection_node")

  def format_human_prompt(
    self, state: ContextRetrievalState, context_list: Sequence[str], search_result: str
  ) -> str:
    context_info = self.HUMAN_PROMPT.format(
      query=state["query"], old_context="\n\n".join(context_list), new_context=search_result
    )
    return context_info

  def __call__(self, state: ContextRetrievalState):
    context_list = state.get("context", [])
    for tool_message in extract_last_tool_messages(state["context_provider_messages"]):
      for search_result in neo4j_data_for_context_generator(tool_message.artifact):
        human_prompt = self.format_human_prompt(state, context_list, search_result)
        response = self.model.invoke({"human_prompt": human_prompt})
        self._logger.debug(f"Is this search result {search_result} relevant?: {response.relevant}")
        if response.relevant:
          context_list.append(search_result)
    self._logger.info(f"Context selection complete, returning context {context_list}")
    return {"context": context_list}
