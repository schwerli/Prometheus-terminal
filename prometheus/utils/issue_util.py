from typing import Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def format_issue_comments(issue_comments: Sequence[Mapping[str, str]]):
  """Formats a sequence of issue comments into a readable string.

  Combines multiple issue comments with their associated usernames into a
  formatted string suitable for inclusion in the response context.

  Args:
    issue_comments: Sequence of mappings containing 'username' and 'comment'
      keys for each issue comment.

  Returns:
    Formatted string containing all comments with usernames, separated by newlines.
  """
  formatted_issue_comments = []
  for issue_comment in issue_comments:
    formatted_issue_comments.append(f"{issue_comment['username']}: {issue_comment['comment']}")
  return "\n\n".join(formatted_issue_comments)


def format_agent_tool_message_history(messages: Sequence[BaseMessage]) -> str:
  formatted_messages = []
  for message in messages:
    if isinstance(message, AIMessage):
      if message.content:
        formatted_messages.append(f"Assistant internal thought: {message.content}")
      if (
        message.additional_kwargs
        and "tool_calls" in message.additional_kwargs
        and message.additional_kwargs["tool_calls"]
      ):
        for tool_call in message.additional_kwargs["tool_calls"]:
          formatted_messages.append(f"Assistant executed tool: {tool_call['function']}")
    elif isinstance(message, ToolMessage):
      formatted_messages.append(f"Tool output: {message.content}")
  return "\n".join(formatted_messages)


def format_test_commands(test_commands: Sequence[str]) -> str:
  test_commands_with_prefix = [f"$ {test_command}" for test_command in test_commands]
  return "\n".join(test_commands_with_prefix)
