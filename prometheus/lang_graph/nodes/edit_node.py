"""Code editing functionality for automated issue resolution and fixing.

This module implements a specialized code editing agent that automatically handles
build failures, test failures, and other code-related issues. It uses a language
model with structured tools to analyze problems and implement precise code changes
while maintaining code integrity.
"""

import functools
import logging
from typing import Dict

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.tools import file_operation
from prometheus.utils.lang_graph_util import truncate_messages


class EditNode:
  SYS_PROMPT = """\
You are a specialized editing agent responsible for implementing precise changes to files. Your
primary focus is on accurately implementing the code changes that have been analyzed and
proposed by the user.

ROLE AND RESPONSIBILITIES:
- Implement the exact code changes specified in the analysis
- Maintain code quality and consistent style
- Ensure precise content replacement
- Verify changes after implementation

THINKING PROCESS:
For each edit operation, follow these steps:
1. READ: Review the current file state using read_file
2. LOCATE: Identify the exact content to be replaced based on the analysis
3. EXECUTE: Make the changes with appropriate tools
4. VALIDATE: Verify the changes by reading the file again

CRITICAL FILE EDIT BEHAVIOR:
The edit_file operation performs an EXACT STRING REPLACEMENT in the file:
- Matches must be exact (including whitespace and indentation)
- Only one match of old_content should exist in the file
- If multiple matches exist, more context is needed
- If no matches exist, content must be verified

EXAMPLES:

<example id="simple-replacement">
<file_before>
def calculate_sum(a: int, b: int) -> int:
    # TODO: Implement addition
    return 0  # Incorrect placeholder

def other_function():
</file_before>

<thought_process>
1. The analysis indicates we need to replace the incorrect return statement
2. Need to replace the exact line "    return 0  # Incorrect placeholder"
3. Verify this string appears exactly once in the file
</thought_process>

<edit_operation>
call edit_file tool with:
old_content="    return 0  # Incorrect placeholder"
new_content="    return a + b  # Implemented correct addition"
</edit_operation>
</example>

<example id="multi-line-replacement">
<file_before>
class StringUtils:
    def reverse_string(self, s: str) -> str:
        # TODO: implement proper reversal
        result = ""
        result += s  # Bug: just copies string
        return result  # Doesn't reverse
    
    def other_method():
</file_before>

<thought_process>
1. The analysis specifies replacing the entire implementation block
2. Need to match the exact block including comments and whitespace
3. Verify this block appears exactly once
</thought_process>

<edit_operation>
call edit_file tool with:
old_content="        # TODO: implement proper reversal
        result = ""
        result += s  # Bug: just copies string
        return result  # Doesn't reverse"
new_content="        result = s[::-1]  # Proper string reversal
        return result"
</edit_operation>
</example>

IMPORTANT REMINDERS:
- You MUST use the provided tools to edit the files
- Always read the file first to get its exact content
- Include all relevant whitespace and indentation in old_content
- When replacing multiple lines, include all of them in old_content
- If multiple matches are found, include more context in old_content
- Verify the uniqueness of the match before making changes
- Focus on implementing the changes exactly as specified in the analysis
"""

  def __init__(self, model: BaseChatModel, kg: KnowledgeGraph):
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.tools = self._init_tools(kg.get_local_path())
    self.model_with_tools = model.bind_tools(self.tools)
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.edit_node")

  def _init_tools(self, root_path: str):
    """Initializes file operation tools with the given root path.

    Args:
      root_path: Base directory path for all file operations.

    Returns:
      List of StructuredTool instances configured for file operations.
    """
    tools = []

    read_file_fn = functools.partial(file_operation.read_file, root_path=root_path)
    read_file_tool = StructuredTool.from_function(
      func=read_file_fn,
      name=file_operation.read_file.__name__,
      description=file_operation.READ_FILE_DESCRIPTION,
      args_schema=file_operation.ReadFileInput,
    )
    tools.append(read_file_tool)

    read_file_with_line_numbers_fn = functools.partial(
      file_operation.read_file_with_line_numbers, root_path=root_path
    )
    read_file_with_line_numbers_tool = StructuredTool.from_function(
      func=read_file_with_line_numbers_fn,
      name=file_operation.read_file_with_line_numbers.__name__,
      description=file_operation.READ_FILE_WITH_LINE_NUMBERS_DESCRIPTION,
      args_schema=file_operation.ReadFileWithLineNumbersInput,
    )
    tools.append(read_file_with_line_numbers_tool)

    create_file_fn = functools.partial(file_operation.create_file, root_path=root_path)
    create_file_tool = StructuredTool.from_function(
      func=create_file_fn,
      name=file_operation.create_file.__name__,
      description=file_operation.CREATE_FILE_DESCRIPTION,
      args_schema=file_operation.CreateFileInput,
    )
    tools.append(create_file_tool)

    delete_fn = functools.partial(file_operation.delete, root_path=root_path)
    delete_tool = StructuredTool.from_function(
      func=delete_fn,
      name=file_operation.delete.__name__,
      description=file_operation.DELETE_DESCRIPTION,
      args_schema=file_operation.DeleteInput,
    )
    tools.append(delete_tool)

    edit_file_fn = functools.partial(file_operation.edit_file, root_path=root_path)
    edit_file_tool = StructuredTool.from_function(
      func=edit_file_fn,
      name=file_operation.edit_file.__name__,
      description=file_operation.EDIT_FILE_DESCRIPTION,
      args_schema=file_operation.EditFileInput,
    )
    tools.append(edit_file_tool)

    return tools

  def __call__(self, state: Dict):
    message_history = [self.system_prompt] + state["edit_messages"]
    truncated_message_history = truncate_messages(message_history)
    response = self.model_with_tools.invoke(truncated_message_history)

    self._logger.debug(f"EditNode response:\n{response}")
    return {"edit_messages": [response]}
