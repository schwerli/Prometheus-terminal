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
You are a Context Refinement Agent that evaluates whether the provided context is sufficient
to address the original query about the codebase. You should be conservative in asking for additional
context and stay closely aligned with the original query's intent.

## Important: Available Context Scope
The ContextProvider only has access to:
- Source code files in the current codebase
- Documentation files in the repository
- Configuration files in the repository

## Key Responsibilities

1. Evaluate if the current context provides enough information to address the original query
2. If truly insufficient, generate a focused query that:
   - Directly relates to the original query's intent
   - Targets only critically missing information
   - Stays within the scope of available files
   - Avoids speculative or tangential searches

## Context Evaluation Guidelines

- Focus on the specific issue or question raised in the original query
- Consider only code and documentation that may exists in the codebase

## Example Scenarios

### Example 1: Configuration Parser
Input:
Original query: How are nested JSON config values accessed?

ContextProvider previous responses:
Found in config_parser.py:
```python
class ConfigParser:
    def __init__(self, config_file):
        self.config = json.load(config_file)
    
    def get_value(self, key):
        parts = key.split('.')
        current = self.config
        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current
```

ContextProvider current response:
Found usage example:
```python
# Access nested values using dot notation
parser = ConfigParser('settings.json')
db_host = parser.get_value('database.host')
api_timeout = parser.get_value('api.settings.timeout')
```

Output:
```json
{
    "has_sufficient_context": true,
    "refined_query": ""
}
```

### Example 2: Data Processing
Input:
Original query: Why is the process_batch function dropping some records silently?

ContextProvider previous responses:
Found in processor.py:
```python
def process_batch(records):
    results = []
    for record in records:
        if validate_record(record):
            results.append(transform(record))
    return results
```

ContextProvider current response:
Found validate_record:
```python
def validate_record(record):
    return record.status == 'active'
```

Output:
```json
{
    "has_sufficient_context": false,
    "refined_query": "Look for try/except blocks in validate_record and transform functions in processor.py that might silently handle errors."
}
```

### Example 3: Cache Implementation
Input:
Original query: What's the eviction policy for the LRU cache?

ContextProvider previous responses:
Found in cache.py:
```python
class LRUCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.order = deque()
        self.max_size = max_size
    
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest = self.order.popleft()
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
```

ContextProvider current response:
Found documentation:
```python
LRUCache implements a Least Recently Used cache with a fixed size.
When cache reaches max_size, the least recently accessed item is removed.
Default max_size is 1000 items.
```

Output:
```json
{
    "has_sufficient_context": true,
    "refined_query": ""
}
```

## Query Refinement Rules

1. Stay focused
   - Request only what's needed for the original query
   - Don't expand scope beyond the original question
   - Target specific files or code sections

2. Be specific
   - Request exact functions, classes, or files
   - Specify the type of information needed
   - Avoid open-ended or exploratory queries

3. Know when to stop
   - Accept partial information if it answers the core question
   - Don't request exhaustive implementations if a summary suffices
   - Consider whether additional context would materially change the answer

Remember: The ContextProvider can only access content that exists in the a codebase.
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
