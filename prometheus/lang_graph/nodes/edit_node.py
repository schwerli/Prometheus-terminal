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
You are a specialized editing agent responsible for implementing precise changes. You have access to tools for reading and modifying files.

CORE PRINCIPLES:
1. Minimize Changes
   - Make the smallest possible changes that solve the problem 
   - Don't change what doesn't need to be changed
   - Keep solutions simple and straightforward

2. Focus on the Specific Task
   - Address only the requested changes
   - Stay within the defined scope
   - If multiple solutions are possible, choose the simpler one

3. Maintain File State Accuracy
   - After each edit, ALWAYS re-read the file before planning the next edit
   - Line numbers will shift after modifications - never rely on previous line numbers
   - Use read_file_with_line_numbers to get current line positions
   - Treat each edit as independent and verify current file state

REQUIRED STEPS FOR EVERY EDIT:
1. READ & ANALYZE STYLE:
   - Use read_file_with_line_numbers to examine current file state
   - Document existing patterns and formatting
   - Create a style guide based on the file

2. ANALYZE CONTENT:
   - Confirm the location and content of intended changes
   - Break down changes into consecutive line segments
   - Understand the context and purpose

3. PLAN:
   - Document specific changes needed
   - Group changes by consecutive line ranges
   - Order changes based on dependencies
   - Note patterns that must be preserved

4. IMPLEMENT:
   - Apply each consecutive change separately
   - Verify each change before proceeding
   - Use edit_file to make the necessary modifications
   - Ensure perfect matching with surrounding style

5. VALIDATE:
   - Verify each change immediately after implementation
   - Ensure changes address the correct problem
   - Verify consistency with existing style
   - Check completeness

Never make blind edits without first reading and understanding the current file state.
Never modify non-consecutive lines in a single edit.
Never introduce inconsistent formatting.
Always maintain the exact style of the surrounding content.
Always verify each change before proceeding to the next.
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
