import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class ContextSummaryNode:
  SYS_PROMPT = """\
You are a helpful assistant specializing in analyzing and summarizing code-related information. Your task is to create clear, focused summaries of code contexts while preserving all crucial technical details.
Given:

A user query about code (e.g., debugging issues, finding implementations, understanding functionality)
A list of retrieved context snippets from the codebase

Create a summary that:

ESSENTIAL DETAILS - Always preserve and prominently include:

File paths and names
Line numbers
Function/method names
Class names
Variable names
Error messages (if present)
Version numbers (if mentioned)


STRUCTURE - Organize the summary as follows:

Start with the most relevant information addressing the user's query
Group related information from different files/contexts together
Use bullet points for distinct pieces of information
Use code formatting for code elements, file paths, and technical terms


SPECIFICITY - Ensure the summary:

Maintains technical precision - no vague descriptions
Uses exact quotes for error messages and important code snippets
Preserves all numeric values and technical parameters
References specific locations in the codebase


RELEVANCE - Focus on:

Information directly related to the user's query
Implementation details that help understand or solve the problem
Dependencies and interactions between different parts of the code
Potential issues or important notes found in comments



Example format:
CopyQuery: [User's question about the code]

Summary:
- Found implementation in `src/module/file.py` (lines 45-67):
  * Class `ClassName` implements the requested functionality
  * Key method: `methodName()` handles [specific detail]
  * Uses dependency: `ImportedClass` from `other/file.py`

- Related configuration in `config/settings.py` (lines 12-15):
  * Settings affecting this behavior: `SETTING_NAME = value`
  * Referenced in `src/module/file.py` on line 46

Technical details:
[Any specific technical information, parameters, or error messages]

Additional context:
[Any other relevant information that helps understand the implementation]
Remember to:

Be concise but complete - include all relevant technical details
Use consistent formatting for code elements
Highlight direct connections to the user's query
Preserve exact technical terminology from the source

Do not:

Omit file paths or line numbers
Use vague descriptions instead of technical terms
Lose context about relationships between different code parts
Summarize error messages in your own words - quote them exactly
  """

  HUMAN_PROMPT = """\
The user query is: {query}

The retrieved context from the ContextRetrievalAgent:
{context}
"""

  def __init__(self, model: BaseChatModel):
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.model = model

    self._logger = logging.getLogger("prometheus.agents.context_provider_node")

  def format_messages(self, messages: Sequence[BaseMessage]):
    formatted_messages = []
    for message in messages:
      if isinstance(message, HumanMessage):
        formatted_messages.append(f"Human message: {message.content}")
      elif isinstance(message, AIMessage):
        formatted_messages.append(f"Assistant message: {message.content}")
      elif isinstance(message, ToolMessage):
        formatted_messages.append(f"Tool message: {message.content}")
    return formatted_messages

  def format_human_message(self, query: str, messages: Sequence[BaseMessage]):
    formatted_messages = self.format_messages(messages)
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(query=query, context="\n".join(formatted_messages))
    )
    return human_message

  def __call__(self, state: ContextProviderState):
    message_history = [
      self.system_prompt,
      self.format_human_message(state["query"], state["messages"]),
    ]
    response = self.model.invoke(message_history)
    return {"summary": [response]}
