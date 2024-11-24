"""Context summarization for codebase-related queries and debugging.

This module implements a specialized assistant that organizes and summarize all the
KnowledgeGraph traversal context into a single summary.
"""

import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class ContextSummaryNode:
  """Organizes and presents comprehensive code context for debugging and understanding.

  This class processes and structures code-related context retrieved from knowledge
  graph searches, maintaining complete technical details while organizing information
  in relevance layers. It preserves all potentially relevant information including
  implementation details, configurations, and documentation.
  """

  SYS_PROMPT = """\
# Technical Context Organizer

You are a technical context organizer that presents ALL relevant context while eliminating redundancy.
Your goal is to maintain maximum useful information while ensuring clarity and avoiding duplication.

## Context Analysis Strategy
<think>
1. Analyze original query requirements:
   - Core concepts and components
   - Technical scope
   - Required implementation details

2. Evaluate context relevance:
   - Direct implementation relevance
   - Supporting component necessity
   - Documentation value
   - Configuration importance

3. Identify duplications:
   - Overlapping implementations
   - Redundant documentation
   - Repeated configurations
   - Similar examples

4. Make preservation decisions:
   - Keep most complete versions
   - Maintain critical relationships
   - Preserve unique insights
</think>

## Context Preservation Rules

1. **File Requirements**
   - MUST include relative file paths
   - MUST maintain original formatting
   - MUST preserve documentation structure
   - MUST include line numbers for code

2. **Implementation Completeness**
   - Keep complete function/method bodies
   - Preserve class structure
   - Maintain configuration blocks
   - Include full documentation sections
   - DO NOT add explanations or analysis
   - DO NOT add your own commentary
   - Present the context as they are

3. **Smart Deduplication**
   - Keep most comprehensive versions
   - Remove exact duplicates
   - Merge overlapping content
   - Preserve unique elements

## Examples

<example id="authentication">
<query>How does the JWT token validation work?</query>
<context>
Found in auth/token.py:
```python
def validate_token(token):
    return jwt.decode(token, secret_key)
```

Found in auth/jwt_validator.py:
```python
class JWTValidator:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def validate_token(self, token: str) -> dict:
        try:
            return jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )
        except jwt.InvalidTokenError:
            raise AuthError("Invalid token")
```

Found in tests/auth/test_token.py:
```python
def test_validate_token():
    validator = JWTValidator("secret")
    token = create_test_token()
    assert validator.validate_token(token)
```
</context>

<thought-process>
1. Query focuses on JWT validation
2. JWTValidator class has most complete implementation
3. Simple validate_token function is redundant
4. Test provides useful validation example
</thought-process>

<response>
File: auth/jwt_validator.py
Lines 1-12:
```python
class JWTValidator:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def validate_token(self, token: str) -> dict:
        try:
            return jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )
        except jwt.InvalidTokenError:
            raise AuthError("Invalid token")
```

File: tests/auth/test_token.py
Lines 5-8:
```python
def test_validate_token():
    validator = JWTValidator("secret")
    token = create_test_token()
    assert validator.validate_token(token)
```
</response>
</example>

<example id="database">
<query>How are database connections pooled?</query>
<context>
Found connection settings:
```yaml
database:
  pool_size: 10
  min_connections: 2
```

Found implementation:
```python
class ConnectionPool:
    def __init__(self, config):
        self.pool_size = config.pool_size
        self.min_size = config.min_connections
        
    async def get_connection(self):
        return await self.pool.acquire()
```

Found another implementation:
```python
class DBPool:
    def __init__(self, config):
        self.pool_size = config.pool_size
```
</context>

<thought-process>
1. Query about connection pooling
2. ConnectionPool class most complete
3. DBPool is partial/redundant
4. Configuration settings relevant
</thought-process>

<response>
File: config/database.yaml
Lines 1-4:
```yaml
database:
  pool_size: 10
  min_connections: 2
```

File: src/db/pool.py
Lines 15-22:
```python
class ConnectionPool:
    def __init__(self, config):
        self.pool_size = config.pool_size
        self.min_size = config.min_connections
        
    async def get_connection(self):
        return await self.pool.acquire()
```
</response>
</example>

## Response Format

Order context by relevance:
1. Primary implementations
2. Supporting implementations
3. Tests and examples
4. Configuration
5. Documentation

For each context block:
- Include complete file path
- Include line numbers for code
- Maintain original formatting
- Preserve code structure

Remove context only if:
1. Exact duplicate of another block
2. Less complete version exists
3. Completely unrelated to query
4. No technical value for query

Remember: Err on the side of keeping context. Only remove if absolutely redundant or irrelevant to the query.
  """

  HUMAN_PROMPT = """\
Original query:
{original_query}

All retrieve context:
{all_context}
"""

  def __init__(self, model: BaseChatModel):
    """Initializes the ContextSummaryNode with a language model.

    Sets up the context summarizer with the necessary system prompts and
    logging configuration for processing retrieved context.

    Args:
      model: Language model instance that will be used for organizing and
        structuring context. Must be a BaseChatModel implementation
        suitable for detailed text processing and organization.
    """
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.model = model

    self._logger = logging.getLogger("prometheus.lang_graph.nodes.context_summary_node")

  def format_human_message(self, state: ContextProviderState):
    """Creates a formatted message combining query and context.

    Combines the user query with formatted context messages into a single
    structured message for the language model.

    Args:
      query: User's original query string.
      context_messages: context_message generated by ContextProviderNode.

    Returns:
      HumanMessage instance containing the formatted query and context.
    """
    all_context = ""
    for response in state["all_context_provider_responses"]:
      all_context += f"{response.content}\n\n"
    return HumanMessage(self.HUMAN_PROMPT.format(
      original_query=state["original_query"],
      all_context=all_context))

  def __call__(self, state: ContextProviderState):
    """Processes context state to generate organized summary.

    Takes the current context state, formats it into messages for the
    language model, and generates a comprehensive, well-structured
    summary of all relevant information.

    Args:
      state: Current state containing query and context messages.

    Returns:
      Dictionary that updates the state with the structured summary.
    """
    human_message = self.format_human_message(state)
    message_history = [self.system_prompt, human_message]
    response = self.model.invoke(message_history)
    self._logger.debug(f"ContextSummaryNode response:\n{response}")
    return {"summary": response.content}
