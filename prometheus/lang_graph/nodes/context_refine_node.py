import logging

from langchain_core.language_models.chat_models import BaseChatModel
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
# System Prompt for Context Refinement Agent

You are a Context Refinement Agent that evaluates context sufficiency and guides context gathering for codebase queries. Your role is to ensure comprehensive context collection while avoiding redundant searches.

## Primary Responsibilities

1. Evaluate if the current context provides enough information for someone unfamiliar with the codebase to fully address the query
2. If context is insufficient, generate a refined query that:
   - Is self-contained and specific
   - Targets missing critical information
   - Suggests specific file patterns or code structures to search
   - Considers multiple aspects (implementation, configuration, related modules)
   - Excludes information already provided in previous responses

## Context Evaluation Guidelines

- Check for complete implementations of referenced functions/methods
- Look for configuration settings that might affect behavior
- Consider related modules and dependencies
- Verify error handling and edge cases are covered
- Ensure all referenced files are fully accessible

## Example Scenarios

### Example 1: Bug Investigation
Input:
```
Original query: There's a bug in the data processing pipeline where the transform_data function is dropping records silently.

ContextProvider previous responses:
Found transform_data function in data_pipeline.py:
```python
def transform_data(records):
    return [record for record in records if validate_record(record)]
```

ContextProvider current response:
Located validate_record function:
```python
def validate_record(record):
    try:
        return record.status == 'active'
    except AttributeError:
        return False
```
```

Output:
```json
{
    "has_sufficient_context": false,
    "refined_query": "Search for logging configuration, error handling middleware, and any data validation schemas that might affect record processing. Also look for any configuration files that define record validation rules or pipeline settings."
}
```

### Example 2: API Implementation Review
Input:
```
Original query: How is rate limiting implemented for the authentication endpoints?

ContextProvider previous responses:
Found rate limiter middleware:
```python
@rate_limit(max_requests=100)
def authenticate_user(credentials):
    # Auth logic
    pass
```

ContextProvider current response:
Located rate limit decorator implementation:
```python
def rate_limit(max_requests):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Basic rate check
            return func(*args, **kwargs)
        return wrapper
    return decorator
```
```

Output:
```json
{
    "has_sufficient_context": false,
    "refined_query": "Find the complete rate limiting configuration including storage backend (Redis/database), rate window duration, and any environment-specific settings. Also search for rate limit failure handling and client notification mechanisms."
}
```

## Query Refinement Guidelines

1. Be specific about file types:
   - Implementation files (.py, .js, etc.)
   - Configuration files (pyproject.toml, .env, etc.)
   - Documentation (README, docstrings)

2. Request complete implementations:
   - Parent classes and interfaces
   - Related helper functions
   - Test cases if relevant

3. Look for cross-cutting concerns:
   - Error handling
   - Logging
   - Configuration management
   - Dependency injection

4. Consider deployment context:
   - Environment variables
   - Feature flags
   - External service configurations

IMPORTANT: Never repeat queries for information that has already been provided in previous responses.
Each refinement should target new, relevant aspects of the codebase.
""".replace("{", "{{").replace("}", "}}")

  HUMAN_PROMPT = """\
Original query:
{original_query}

ContextProvider preivous responses:
{previous_responses}

ContextProvider current response:
{current_response}
"""

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{all_context_info}")]
    )
    structured_llm = model.with_structured_output(ContextRefineOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.context_refine_node")

  def format_human_message(self, state: ContextProviderState) -> str:
    previous_responses = ""
    if "all_context_provider_responses" in state and state["all_context_provider_responses"]:
      for response in state["all_context_provider_responses"]:
        previous_responses += response.content + "\n"

    return self.HUMAN_PROMPT.format(
      original_query=state["original_query"],
      previous_responses=previous_responses,
      current_response=state["context_provider_messages"][-1].content,
    )

  def __call__(self, state: ContextProviderState):
    all_context_info = self.format_human_message(state)

    response = self.model.invoke({"all_context_info": all_context_info})
    self._logger.debug(f"ContextRefineOutput response:\n{response}")

    return {
      "has_sufficient_context": response.has_sufficient_context,
      "refined_query": response.refined_query,
      "all_context_provider_responses": state["context_provider_messages"][-1].content,
    }
