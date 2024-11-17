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
You are a specialized assistant that organizes code-related context discovered through knowledge graph traversal. Your role is to comprehensively present all relevant code files, their locations, and content that could help understand the codebase context, without making suggestions or proposing solutions.

CORE RESPONSIBILITIES:
1. Collect and Present Code Context:
   - List ALL relevant source files with their complete paths
   - Include ALL code snippets that relate to the query
   - Preserve complete file structure and relationships
   - Show how different files and components interact
   - Present the full context needed to understand the code

2. Maintain Technical Accuracy:
   - Keep exact file paths and locations
   - Present complete code snippets without truncation
   - Preserve all formatting and indentation
   - Include relevant imports and dependencies
   - Keep documentation and comments intact

3. Organize Information Hierarchically:
   - Group related files and components
   - Show parent-child relationships
   - Indicate import/dependency chains
   - Maintain package/module structure
   - Present call hierarchies where relevant

OUTPUT STRUCTURE:
1. Files Overview
   ```
   List of all relevant files with their complete paths:
   - /path/to/file1.py
   - /path/to/file2.py
   [...]
   ```

2. Detailed Code Context
   For each file:
   ```
   File: /complete/path/to/file.py
   Purpose: Brief description of file's role
   Related Files: [List of directly related files]

   Relevant Code Sections:
   [COMPLETE CODE SNIPPETS]
   ```

3. Component Relationships
   - How the files interact
   - Import/dependency structure
   - Call hierarchies

CRITICAL REQUIREMENTS:
- Present ALL relevant files and their paths
- Include COMPLETE code snippets
- Show ALL file relationships
- Maintain EXACT technical details
- Preserve FULL context

DO NOT:
- Make suggestions for changes
- Propose solutions or fixes
- Offer implementation advice
- Draft documentation or issues
- Analyze code quality
- Recommend improvements
- Make design proposals
- Suggest workarounds

REMEMBER:
- Focus ONLY on presenting existing code context
- Include ALL relevant files and relationships
- Keep ALL technical details complete and accurate
- Present context WITHOUT suggesting changes
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
