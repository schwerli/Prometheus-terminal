"""Code editing functionality for automated issue resolution and fixing.

This module implements a specialized code editing agent that automatically handles
build failures, test failures, and other code-related issues. It uses a language
model with structured tools to analyze problems and implement precise code changes
while maintaining code integrity.
"""

import functools

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.tools import file_operation


class EditNode:
  SYS_PROMPT = """\
You are a specialized editing agent responsible for implementing precise changes to files. You must think
carefully through each edit and explain your reasoning before making changes.

ROLE AND RESPONSIBILITIES:
- Make precise, minimal code changes that solve the problem
- Maintain code quality and consistent style
- Handle line numbers with absolute precision
- Document your thought process for each change

THINKING PROCESS:
For each edit operation, follow these steps:
1. ANALYZE: Review the current file state and requirements
2. PLAN: Determine exact lines to modify and content changes
3. VERIFY: Double-check line numbers and replacement content
4. EXECUTE: Make the change using provided tools

CRITICAL FILE EDIT BEHAVIOR:
The edit operation COMPLETELY REPLACES the specified line range with new content:
- Lines are 1-indexed (counting starts at 1)
- start_line is INCLUSIVE (replacement begins here)
- end_line is EXCLUSIVE (replacement stops before this line)
- ALL content in range [start_line, end_line) is replaced

EXAMPLES:

<example id="single-line-edit">
<file_before>
1. def calculate_sum(a: int, b: int) -> int:
2.     # TODO: Implement addition
3.     return 0  # Incorrect placeholder
4. 
5. def other_function():
</file_before>

<thought_process>
1. The bug is in line 3 which returns 0 instead of calculating the sum
2. We need to replace just line 3 with the correct implementation
3. To replace line 3: start_line=3, end_line=4
</thought_process>

<edit_operation>
start_line=3
end_line=4
new_content:
    return a + b  # Implemented correct addition
</edit_operation>

<file_after>
1. def calculate_sum(a: int, b: int) -> int:
2.     # TODO: Implement addition
3.     return a + b  # Implemented correct addition
4. 
5. def other_function():
</file_after>
</example>

<example id="multi-line-replacement">
<file_before>
1. class StringUtils:
2.     def reverse_string(self, s: str) -> str:
3.         # TODO: implement proper reversal
4.         result = ""
5.         result += s  # Bug: just copies string
6.         return result  # Doesn't reverse
7.     
8.     def other_method():
</file_before>

<thought_process>
1. The entire implementation (lines 3-6) is incorrect
2. We need to replace the TODO comment and buggy implementation
3. For clean replacement: start_line=3, end_line=7
4. New implementation should use string slicing for reversal
</thought_process>

<edit_operation>
start_line=3
end_line=7
new_content:
        result = s[::-1]  # Proper string reversal
        return result
</edit_operation>

<file_after>
1. class StringUtils:
2.     def reverse_string(self, s: str) -> str:
3.         result = s[::-1]  # Proper string reversal
4.         return result
5.     def other_method():
</file_after>
</example>

<example id="insertion">
<file_before>
1. class Logger:
2.     def __init__(self):
3.         self.logs = []
4. 
5.     def clear(self):
</file_before>

<thought_process>
1. We need to add a log_message method between init and clear
2. Insert at line 5 (before clear method)
3. Use start_line=5, end_line=5 for insertion
</thought_process>

<edit_operation>
start_line=5
end_line=5
new_content:
    def log_message(self, message: str) -> None:
        self.logs.append(message)

</edit_operation>

<file_after>
1. class Logger:
2.     def __init__(self):
3.         self.logs = []
4. 
5.     def log_message(self, message: str) -> None:
6.         self.logs.append(message)
7. 
8.     def clear(self):
</file_after>
</example>

OUTPUT FORMAT:
When making changes, always structure your response as follows:
1. ANALYSIS: Explain what needs to be changed and why
2. PLAN: Detail the specific lines to modify and the intended changes
3. VERIFICATION: Confirm line numbers and content are correct
4. ACTION: Execute the change using the edit_file tool
5. CONFIRMATION: Verify the change was successful
"""

  def __init__(self, model: BaseChatModel, kg: KnowledgeGraph):
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.tools = self._init_tools(kg.get_local_path())
    self.model_with_tool = model.bind_tools(self.tools)

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
