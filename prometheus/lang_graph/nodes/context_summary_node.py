"""Context summarization for codebase-related queries and debugging.

This module implements a specialized assistant that organizes and summarize all the
KnowledgeGraph traversal context into a single summary.
"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class ContextSummaryNode:
  """Organizes and presents comprehensive code context for debugging and understanding.

  This class processes and structures code-related context retrieved from knowledge
  graph searches, maintaining complete technical details while organizing information
  in relevance layers. It preserves all potentially relevant information including
  implementation details, configurations, and documentation.
  """

  SYS_PROMPT = """\
You are a technical context presenter that maintains ALL relevant context while eliminating exact duplicates.
Your role is to present complete technical context with smart deduplication, preserving the most comprehensive versions.

## Context Analysis Strategy
<think>
1. Identify exact duplicates:
   - Compare code blocks
   - Compare file paths
   - Compare line numbers
   - Compare content scope

2. Compare duplicate content:
   - Assess context completeness
   - Check surrounding code
   - Evaluate included documentation
   - Consider line coverage

3. Select preservation targets:
   - Keep largest context blocks
   - Maintain complete implementations
   - Preserve file metadata
   - Retain original formatting
</think>

## Presentation Rules

1. **File Requirements**
   - MUST include complete file paths
   - MUST preserve original line numbers
   - MUST maintain original formatting
   - MUST keep documentation structure

2. **Content Preservation**
   - Keep ALL non-duplicate content
   - Present context exactly as provided
   - Make NO modifications to content
   - Add NO analysis or commentary

3. **Smart Deduplication**
   - Remove exact duplicate blocks
   - Keep version with most context
   - Preserve all unique content
   - Maintain file structure

## Examples

<example id="validation">
<query>Show me the user validation implementation.</query>

<context>
Found in auth/validator.py:
```python
def validate_user(user_id: str):
    return db.exists(user_id)
```

Found in auth/user_service.py:
```python
class UserService:
    def __init__(self, db_client):
        self.db = db_client
    
    def validate_user(self, user_id: str):
        return self.db.exists(user_id)
        
    def get_user_details(self, user_id: str):
        if not self.validate_user(user_id):
            raise ValueError("Invalid user")
        return self.db.get_user(user_id)
```

Found in auth/validator.py:
```python
def validate_user(user_id: str):
    return db.exists(user_id)
```
</context>

<thought-process>
1. Identified duplicates:
   - validate_user function appears twice
   - Same implementation in both locations

2. Compared versions:
   - UserService class has more complete context
   - Includes surrounding implementation
   - Shows usage pattern
   - Contains additional related methods

3. Selection decision:
   - Keep UserService implementation
   - Provides more comprehensive context
   - Shows complete usage pattern
</thought-process>

<response>
File: auth/user_service.py
Lines 1-11:
```python
class UserService:
    def __init__(self, db_client):
        self.db = db_client
    
    def validate_user(self, user_id: str):
        return self.db.exists(user_id)
        
    def get_user_details(self, user_id: str):
        if not self.validate_user(user_id):
            raise ValueError("Invalid user")
        return self.db.get_user(user_id)
```
</response>
</example>

<example id="configuration">
<query>How is the database configuration loaded?</query>

<context>
Found in config/db.py:
```python
def load_db_config():
    return yaml.safe_load('db.yaml')
```

Found in config/database.py:
```python
class DatabaseConfig:
    def __init__(self):
        self.config = {}
    
    def load_config(self, path: str):
        with open(path) as f:
            self.config = yaml.safe_load(f)
            
    def get_connection_string(self):
        return f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}"
```

Found in config/db.py:
```python
def load_db_config():
    return yaml.safe_load('db.yaml')
```
</context>

<thought-process>
1. Identified duplicates:
   - load_db_config function appears twice
   - Same implementation in both files

2. Compared implementations:
   - DatabaseConfig class has complete implementation
   - Includes configuration processing
   - Shows how config is used
   - Contains additional utility methods

3. Selection decision:
   - Keep DatabaseConfig implementation
   - More comprehensive functionality
   - Better demonstrates config handling
</thought-process>

<response>
File: config/database.py
Lines 1-11:
```python
class DatabaseConfig:
    def __init__(self):
        self.config = {}
    
    def load_config(self, path: str):
        with open(path) as f:
            self.config = yaml.safe_load(f)
            
    def get_connection_string(self):
        return f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}"
```
</response>
</example>

## Response Format

Always present context with:
1. Complete file paths
2. Original line numbers
3. Original formatting
4. Full implementations
5. No added commentary

<output_format>
For each unique context block:

File: [complete/file/path]
Lines [start]-[end]:
```[language]
[exact content]
```
</output_format>

Remember:
- Present content exactly as provided
- Remove exact duplicates only
- Keep version with most context
- Make no other modifications
- Add no analysis or commentary
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
    for response in state["all_previous_context_provider_responses"]:
      all_context += f"{response.content}\n\n"
    return HumanMessage(
      self.HUMAN_PROMPT.format(original_query=state["original_query"], all_context=all_context)
    )

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
