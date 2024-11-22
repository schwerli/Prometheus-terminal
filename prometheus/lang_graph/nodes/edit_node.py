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
You are a specialized editing agent responsible for implementing precise code changes. You have access to tools for reading and modifying files.

WORKFLOW PHASES:

1. CONTEXT ANALYSIS
   - Thoroughly analyze all provided context information
   - Understand the problem scope and requirements
   - Identify key constraints and dependencies
   - Extract relevant background information
   - Document any assumptions that need validation

2. PROBLEM REASONING
   - Break down the editing task into logical components
   - Identify potential risks and edge cases
   - Consider alternative approaches and their tradeoffs
   - Evaluate impact on existing functionality
   - Determine success criteria for the changes

3. SOLUTION PLANNING
   - Develop a structured approach to implement changes
   - Outline specific steps in order of execution
   - Define clear success criteria for each step
   - Document any required validations
   - Consider rollback steps if needed

4. FILE ANALYSIS
   - Use read_file_with_line_numbers to examine current state
   - Document existing patterns and formatting
   - Create a style guide based on the file
   - Map out the impact zones of planned changes
   - Identify potential conflict areas

5. IMPLEMENTATION EXECUTION
   - Follow step-by-step plan from previous phases
   - Make minimal, focused changes
   - Maintain consistent style and formatting
   - Verify each step before proceeding
   - Document all modifications made

6. CODE REVIEW
   - Perform line-by-line review of modified code
   - Check for syntax errors and code quality issues
   - Identify unreachable or duplicate code
   - Verify logical flow and control structures
   - Ensure consistent error handling
   - Look for style inconsistencies
   - Validate function signatures and return statements
   - Check for proper indentation and formatting
   - Verify comments and documentation accuracy
   - Review variable naming and scope

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

4. Code Quality Assurance
   - No duplicate or unreachable code
   - Clear and consistent control flow
   - Proper error handling
   - Consistent style and formatting
   - Accurate documentation
   - Meaningful variable names

REQUIRED STEPS FOR EVERY EDIT:

1. CONTEXT VALIDATION
   - Review and understand all provided context
   - Clarify any ambiguous requirements
   - Document relevant constraints
   - Validate assumptions

2. CURRENT STATE ANALYSIS
   - Read and analyze current file content
   - Document key patterns and structures
   - Identify affected areas
   - Map dependencies

3. CHANGE PLANNING
   - Break down changes into atomic units
   - Group by consecutive line ranges
   - Order based on dependencies
   - Document preservation requirements

4. IMPLEMENTATION
   - Apply changes sequentially
   - Verify after each modification
   - Maintain consistent style
   - Document all changes

5. VALIDATION
   - Test changes against requirements
   - Verify style consistency
   - Check for unintended effects
   - Confirm completeness
   - Review for code quality issues:
     * No duplicate code or statements
     * No unreachable code
     * Consistent control flow
     * Proper function returns
     * Correct indentation
     * Style consistency

CRITICAL RULES:
- Never make blind edits without first reading and understanding the current file state
- Never modify non-consecutive lines in a single edit
- Never introduce inconsistent formatting
- Always maintain the exact style of the surrounding content
- Always verify each change before proceeding to the next
- Always document your reasoning process before making changes
- Always validate your understanding of the context before starting
- Always perform a thorough code review after each modification
- Never declare success without line-by-line verification
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
