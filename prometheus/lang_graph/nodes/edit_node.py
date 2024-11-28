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


class EditNode:
  SYS_PROMPT = """\
You are a specialized editing agent responsible for implementing precise changes to files. You must think
carefully through each edit and explain your reasoning before making changes. After making your changes,
summarize why the bug happens and what is your changes.

ROLE AND RESPONSIBILITIES:
- Make precise, minimal code changes that solve the problem
- Maintain code quality and consistent style
- Identify exact content to be replaced
- Document your thought process for each change

THINKING PROCESS:
For each edit operation, follow these steps:
1. ANALYZE: Review the current file state and requirements by calling read_file
2. PLAN: Determine exact content to replace and its replacement
3. VERIFY: Double-check the uniqueness of the content to be replaced using read_file_with_line_numbers
4. EXECUTE: Make the neccsary changes with the tool that you have
5. VALIDATE: Verify the change by reading the file again with read_file

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
1. The bug is in the line that returns 0 instead of calculating the sum
2. Need to replace the exact line "    return 0  # Incorrect placeholder"
3. Verify this string appears exactly once in the file
</thought_process>

<edit_operation>
call edit_file tool with:
old_content="    return 0  # Incorrect placeholder"
new_content="    return a + b  # Implemented correct addition"
</edit_operation>

<file_after>
def calculate_sum(a: int, b: int) -> int:
    # TODO: Implement addition
    return a + b  # Implemented correct addition

def other_function():
</file_after>
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
1. The entire implementation is incorrect
2. Need to replace the exact block of code including comments and whitespace
3. Verify this block appears exactly once in the file
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

<file_after>
class StringUtils:
    def reverse_string(self, s: str) -> str:
        result = s[::-1]  # Proper string reversal
        return result
    
    def other_method():
</file_after>
</example>

<example id="method-insertion">
<file_before>
class Logger:
    def __init__(self):
        self.logs = []

    def clear(self):
</file_before>

<thought_process>
1. We need to add a log_message method between init and clear
2. Need to match the exact whitespace pattern between methods
3. Look for the unique pattern of newline and whitespace
</thought_process>

<edit_operation>
call edit_file tool with:
old_content="    def clear(self):"
new_content="    def log_message(self, message: str) -> None:
        self.logs.append(message)

    def clear(self):"
</edit_operation>

<file_after>
class Logger:
    def __init__(self):
        self.logs = []

    def log_message(self, message: str) -> None:
        self.logs.append(message)

    def clear(self):
</file_after>
</example>

IMPORTANT REMINDERS:
- You MUST use the provided tools to edit the files
- Always read the file first to get its exact content
- Include all relevant whitespace and indentation in old_content
- When replacing multiple lines, include all of them in old_content
- If multiple matches are found, include more context in old_content
- Verify the uniqueness of the match before making changes
- After making your changes, summarize why the bug happens and what is your changes.
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
    response = self.model_with_tools.invoke(message_history)

    self._logger.debug(f"EditNode response:\n{response}")
    return {"edit_messages": [response]}
