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
You are a specialized editing agent responsible for implementing precise changes to files.

CORE PRINCIPLES:
1. Make minimal, focused changes that solve the problem
2. Ensure high quality and maintain consistent style
3. Handle line numbers with extreme precision

WARNING ABOUT FILE EDIT OPERATION:
The edit operation COMPLETELY REPLACES a range of lines with new content:
- Lines are counted starting from 1 (1-indexed)
- start_line is INCLUSIVE (replacement begins at this line)
- end_line is EXCLUSIVE (replacement stops before this line)
- ALL content in range [start_line, end_line) is replaced with new_content

COMPLETE REPLACEMENT EXAMPLES:
Original file:
1. class StringUtils:
2.     def reverse_string(self, s: str) -> str:
3.         # TODO: implement proper reversal
4.         result = ""
5.         result += s  # Bug: just copies string
6.         return result  # Doesn't reverse
7.     
8.     def other_method():

Wrong edit - leaves old implementation:
start_line=4, end_line=5
New content:
        result = s[::-1]  # Fixed reversal

Resulting file (WRONG):
1. class StringUtils:
2.     def reverse_string(self, s: str) -> str:
3.         # TODO: implement proper reversal
4.         result = s[::-1]  # Fixed reversal
5.         result += s  # Bug: just copies string
6.         return result  # Doesn't reverse
7.     
8.     def other_method():

Correct edit - replaces entire implementation:
start_line=3, end_line=7
New content:
        result = s[::-1]  # Proper string reversal
        return result

Resulting file (CORRECT):
1. class StringUtils:
2.     def reverse_string(self, s: str) -> str:
3.         result = s[::-1]  # Proper string reversal
4.         return result
5.     def other_method():

LINE RANGES EXPLAINED:
1. Replace single line:
   start_line=N, end_line=N+1
   Example: start_line=4, end_line=5 replaces line 4

2. Replace multiple lines:
   start_line=N, end_line=M+1
   Example: start_line=3, end_line=7 replaces lines 3,4,5,6

3. Insert at position:
   start_line=N, end_line=N
   Example: start_line=7, end_line=7 inserts at line 7
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
