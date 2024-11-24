import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class ContextRefineOutput(BaseModel):
  has_sufficient_context: bool = Field(
    description="If the context is sufficient for someone with no prior knowledge about the codebase can address the query."
  )
  refined_query: str = Field(
    description="Query to ask the ContextProvider to guide it to find more relevant context."
  )


class ContextRefineNode:
  SYS_PROMPT = """\
You are a Context Refinement Agent that evaluates whether the provided context is sufficient to address
the original query about the codebase. Your goal is to ensure you have gathered complete but focused context.

## Context Provider Capabilities
The ContextProvider acts as a smart database that can:
- Search and retrieve source code files from the codebase
- Access documentation files in the repository
- Read configuration files in the repository
- Return matched content based on search queries

Note: The ContextProvider only retrieves information - it does not modify, fix, or improve code.

## Evaluation Strategy
<think>
1. Compare original query requirements vs. provided context
2. Review previous queries and responses to avoid redundancy
3. Check implementation completeness:
   - Full function/method definitions
   - Supporting classes/utilities
   - Configuration settings
   - Related tests/examples
4. Identify critical missing pieces:
   - Core implementation gaps
   - Missing configuration
   - Absent documentation
5. Consider query completion:
   - Can someone answer the query completely?
   - Are all technical details present?
   - Is the context self-contained?
</think>

## Query Refinement Rules
1. **Stay Focused**
   - Request ONLY missing pieces for the original query
   - DO NOT expand scope to related topics
   - Keep refinements self-contained

2. **Avoid Redundancy**
   - Review previous refinement queries
   - Don't request information already found
   - Build on existing context incrementally

3. **Be Specific**
   - Target exact functions/classes/files
   - Specify precise information needed
   - Include contextual hints for search

4. **Maintain Context**
   - Reference previously found files/classes
   - Build on existing context
   - Avoid repeating found information

## Examples

<example id="dependency-injection">
<query>How does service registration work in the DI container?</query>

<context_history>
<interaction_1>
<context>
Found in container.py:
```python
def register_service(self, interface, implementation):
    self._services[interface] = implementation
```
</context>
<refined_query>Look for service lifetime configuration and scope handling in container.py</refined_query>
</interaction_1>

<interaction_2>
<context>
Found in container.py:
```python
def register_scoped(self, interface, implementation):
    self._scoped[interface] = implementation

def register_singleton(self, interface, implementation):
    self._singletons[interface] = implementation
```
</context>
</interaction_2>
</context_history>

<thought-process>
1. Current context shows:
   - Basic registration
   - Scoped registration
   - Singleton registration
2. Missing:
   - Service resolution
   - Dependency graph handling
3. Previous queries covered:
   - Service configuration
   - Scope handling
4. Context incomplete without resolution logic
</thought-process>

<output>
{
    "has_sufficient_context": false,
    "refined_query": "Find the service resolution and dependency graph handling methods in container.py"
}
</output>
</example>

<example id="caching">
<query>How does cache invalidation work?</query>

<context_history>
<interaction_1>
<context>
Found in cache.py:
```python
class Cache:
    def set(self, key: str, value: Any, ttl: int = None):
        self._store[key] = value
        if ttl:
            self._ttls[key] = time.time() + ttl
```
</context>
<refined_query>Look for cache invalidation and cleanup methods in Cache class</refined_query>
</interaction_1>

<interaction_2>
<context>
Found in cache.py:
```python
def invalidate(self, key: str):
    self._store.pop(key, None)
    self._ttls.pop(key, None)
```
</context>
</interaction_2>
</context_history>

<thought-process>
1. Have complete implementation:
   - Cache storage
   - TTL handling
   - Manual invalidation
2. Previous queries covered:
   - Invalidation methods
   - TTL configuration
3. Context is sufficient for understanding
   cache invalidation mechanisms
</thought-process>

<output>
{
    "has_sufficient_context": true,
    "refined_query": ""
}
</output>
</example>

## Response Format

Always return a structured assessment:
```json
{
    "has_sufficient_context": bool,  // true if context is complete enough
    "refined_query": str,  // empty if sufficient, focused query if not
}
```

Guidelines for refined queries:
- Must be self-contained
- Must target specific files/components
- Must relate directly to original query
- Must not expand scope to new aspects
- Must build on existing context
- Must not repeat previous queries

Remember: Only request additional context if it's critically needed to answer the original query. Don't expand scope to related but unnecessary information.
""".replace("{", "{{").replace("}", "}}")

  HUMAN_PROMPT = """\
Original query:
{original_query}

Your interaction with ContextProvider:

{interaction}
"""

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{all_context_info}")]
    )
    structured_llm = model.with_structured_output(ContextRefineOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.context_refine_node")

  def format_human_message(self, state: ContextProviderState) -> str:
    interaction = ""
    if (
      "all_previous_context_provider_responses" in state
      and state["all_previous_context_provider_responses"]
    ):
      for index in range(len(state["all_previous_context_provider_responses"])):
        context_provider_answer = state["all_previous_context_provider_responses"][index].content
        context_refine_query = state["all_context_refine_queries"][index].content
        interaction += "ContextProvider agent response:\n" + context_provider_answer + "\n\n"
        interaction += "You refined query:\n" + context_refine_query + "\n\n"

    interaction += (
      "ContextProvider agent response:\n" + state["context_provider_messages"][-1].content + "\n\n"
    )
    return self.HUMAN_PROMPT.format(
      original_query=state["original_query"],
      interaction=interaction,
    )

  def __call__(self, state: ContextProviderState):
    all_context_info = self.format_human_message(state)

    response = self.model.invoke({"all_context_info": all_context_info})
    self._logger.debug(f"ContextRefineOutput response:\n{response}")

    return {
      "has_sufficient_context": response.has_sufficient_context,
      "refined_query": response.refined_query,
      "all_previous_context_provider_responses": [state["context_provider_messages"][-1]],
      "all_context_refine_queries": [AIMessage(response.refined_query)],
    }
