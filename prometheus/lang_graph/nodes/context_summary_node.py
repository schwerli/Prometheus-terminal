"""Context summarization for codebase-related queries and debugging.

This module implements a specialized assistant that organizes and summarize all the
KnowledgeGraph traversal context into a single summary.
"""

import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class ContextSummaryNode:
  """Organizes and presents comprehensive code context for debugging and understanding.

  This class processes and structures code-related context retrieved from knowledge
  graph searches, maintaining complete technical details while organizing information
  in relevance layers. It preserves all potentially relevant information including
  implementation details, configurations, and documentation.
  """

  SYS_PROMPT = """\
You are a specialized assistant that summarizes code context discovered through knowledge graph traversal. Your role is to present relevant code context based on the query, focusing primarily on actual implementations while acknowledging example or test code when specifically relevant.

CORE RESPONSIBILITIES:
1. Present Code Context:
   - List relevant source files with their complete paths
   - Include actual code snippets from the codebase
   - Present file structure and relationships as implemented
   - Show component interactions from the code
   - Focus on implementation details that match the query context

2. Maintain Technical Accuracy:
   - Present exact file paths as they exist
   - Show complete code snippets without modification
   - Preserve actual formatting and indentation
   - Document imports and dependencies
   - Include relevant features and functionality

3. Reflect Code Structure:
   - Show actual package organization
   - Document dependency relationships
   - Present module hierarchy
   - Map component interactions
   - Reflect service architecture

OUTPUT STRUCTURE:
1. Files Overview
   ```
   Relevant files with complete paths:
   - /path/to/file1.py 
   - /path/to/file2.py
   [...]
   ```

2. Implementation Details
   For each file:
   ```
   File: /complete/path/to/file.py
   Role: File's function in the system
   Dependencies: [List of direct dependencies]

   Implementation:
   [UNMODIFIED CODE SNIPPETS]
   ```

3. System Architecture
   - Component integration points
   - Dependency structure
   - Call patterns
   - Service relationships

CRITICAL REQUIREMENTS:
- Present relevant code as-is
- Show actual implementation details
- Document system structure
- Maintain technical accuracy
- Focus on query-relevant code

DO NOT:
- Propose code changes
- Suggest improvements
- Offer implementation advice
- Make design proposals
- Analyze code quality
- Recommend fixes
- Draft solutions
- Suggest workarounds

ESSENTIAL FOCUS:
- Summarize code that addresses the query
- Present relevant codebase structure
- Show pertinent implementation details
- Document system architecture
- Describe implemented features
  """

  HUMAN_PROMPT = """\
The user query is: {query}

The retrieved context from another agent:
{context}
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

  def format_messages(self, context_messages: Sequence[BaseMessage]):
    """Formats a sequence of messages into a structured list.

    Converts different types of messages (Human, AI, Tool) into a consistently
    formatted list of strings, preserving the message source and content.

    Args:
      context_messages: Sequence of BaseMessage instances to be formatted.
        Can include HumanMessage, AIMessage, and ToolMessage types.

    Returns:
      List of formatted message strings, each prefixed with its source type.
    """
    formatted_messages = []
    for message in context_messages:
      if isinstance(message, HumanMessage):
        formatted_messages.append(f"Human message: {message.content}")
      elif isinstance(message, AIMessage):
        formatted_messages.append(f"Assistant message: {message.content}")
      elif isinstance(message, ToolMessage):
        formatted_messages.append(f"Tool message: {message.content}")
    return formatted_messages

  def format_human_message(self, query: str, context_messages: Sequence[BaseMessage]):
    """Creates a formatted message combining query and context.

    Combines the user query with formatted context messages into a single
    structured message for the language model.

    Args:
      query: User's original query string.
      context_messages: context_message generated by ContextProviderNode.

    Returns:
      HumanMessage instance containing the formatted query and context.
    """
    formatted_context_messages = self.format_messages(context_messages)
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(query=query, context="\n".join(formatted_context_messages))
    )
    return human_message

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
    message_history = [
      self.system_prompt,
      self.format_human_message(state["query"], state["context_messages"]),
    ]
    response = self.model.invoke(message_history)
    self._logger.debug(f"ContextSummaryNode response:\n{response}")
    return {"summary": response.content}
