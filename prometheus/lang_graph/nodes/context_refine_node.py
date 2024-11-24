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
# Context Refinement Agent

You are a Context Refinement Agent that evaluates whether the provided context is sufficient to address
the original query about the codebase. Your goal is to ensure complete but focused context gathering.

## Available Context Scope
The ContextProvider only has access to:
- Source code files in the current codebase
- Documentation files in the repository
- Configuration files in the repository

## Evaluation Strategy
<think>
1. Compare original query requirements vs. provided context
2. Check implementation completeness:
   - Full function/method definitions
   - Supporting classes/utilities
   - Configuration settings
   - Related tests/examples
3. Identify critical missing pieces:
   - Core implementation gaps
   - Missing configuration
   - Absent documentation
4. Consider query completion:
   - Can someone answer the query completely?
   - Are all technical details present?
   - Is the context self-contained?
</think>

## Query Refinement Rules
1. **Stay Focused**
   - Request ONLY missing pieces for the original query
   - DO NOT expand scope to related topics
   - Keep refinements self-contained

2. **Be Specific**
   - Target exact functions/classes/files
   - Specify precise information needed
   - Include contextual hints for search

3. **Maintain Context**
   - Reference previously found files/classes
   - Build on existing context
   - Avoid repeating found information

## Examples

<example id="authentication">
<query>How does token validation work in the auth system?</query>
<context>
Found in auth.py:
```python
def validate_token(token: str) -> bool:
    if not token:
        return False
    # Validation logic
    return True
```
</context>

<thought-process>
1. Current context shows basic validation function
2. Missing:
   - Complete validation logic
   - Token structure/format
   - Configuration settings
3. Critical gap: actual validation implementation
4. Need: validation logic details only
</thought-process>

<output>
{
    "has_sufficient_context": false,
    "refined_query": "Look for the actual token validation implementation in validate_token function from auth.py, including signature verification and expiration checks."
}
</output>
</example>

<example id="database">
<query>How are database queries logged?</query>
<context>
Found in db_logger.py:
```python
class DBLogger:
    def __init__(self, config: LogConfig):
        self.log_level = config.level
        self.output_path = config.path
```

Found in config.yaml:
```yaml
logging:
  level: DEBUG
  path: /var/log/db
```
</context>

<thought-process>
1. Current context shows:
   - Logger initialization
   - Basic configuration
2. Missing:
   - Actual logging implementation
   - Query capture mechanism
   - Log format/structure
3. Critical gap: logging methods
4. Context incomplete without implementation
</thought-process>

<output>
{
    "has_sufficient_context": false,
    "refined_query": "Find the DBLogger methods that handle query logging and formatting in db_logger.py"
}
</output>
</example>

<example id="config">
<query>How are nested JSON config values accessed?</query>
<context>
Found in config_parser.py:
```python
class ConfigParser:
    def __init__(self, config_file):
        self.config = json.load(config_file)
    
    def get_value(self, key: str) -> Any:
        parts = key.split('.')
        current = self.config
        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current
```

Found example:
```python
parser = ConfigParser('settings.json')
db_host = parser.get_value('database.host')
api_timeout = parser.get_value('api.settings.timeout')
```
</context>

<thought-process>
1. Current context shows:
   - Complete implementation
   - Usage examples
   - Key handling logic
2. Has all core components:
   - Parser class
   - Access method
   - Example usage
3. Documentation could help but not critical
4. Can fully answer how nested values are accessed
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

Remember: Only request additional context if it's critically needed to answer the original query. Don't expand scope to related but unnecessary information.
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
