"""Code editing functionality for automated issue resolution and fixing.

This module implements a specialized code editing agent that automatically handles
build failures, test failures, and other code-related issues. It uses a language
model with structured tools to analyze problems and implement precise code changes
while maintaining code integrity.
"""

import functools
import logging
from typing import Mapping, Sequence

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from prometheus.tools import file_operation


class CodeEditingNode:
  """Implements automated code editing for issue resolution and fixes.

  This class provides functionality to automatically analyze and fix code issues,
  including build failures and test failures. It uses a language model with
  structured tools to make precise code modifications while maintaining code
  integrity and existing patterns.
  """

  SYS_PROMPT = """\
You are a specialized code editing agent responsible for implementing precise code changes to fix build failures, test failures, and issues. You will receive:
1. Issue information (title, description, and comments)
2. Context summary from previously retrieved codebase information
3. Build status and failure logs (if available and enabled)
4. Test status and failure logs (if available and enabled)
5. Access to tools for reading and modifying source code

CORE RESPONSIBILITIES AND WORKFLOW:
1. Initial File Reading (MANDATORY)
   - ALWAYS read the target file(s) using read_file or read_file_with_line_numbers before making any changes
   - Review the current code structure and implementation
   - Document the current state and identify the specific sections that need modification

2. Priority-based Analysis
   a) Build Status Handling
      - If build is enabled (run_build=True) and exists (exist_build=True):
        * If build_fail_log is present, analyze and fix build errors
        * Focus on syntax errors, missing dependencies, or compilation issues
      - If build status is unknown (build disabled or non-existent):
        * Proceed with caution
        * Make changes that maintain basic build integrity
   
   b) Test Status Handling
      - If testing is enabled (run_test=True) and exists (exist_test=True):
        * If test_fail_log is present, analyze test failures carefully
        * Determine if failures are due to:
          - Actual code bugs (fix these)
          - Test bugs (avoid changing these)
          - New tests for issue verification (preserve these)
        * Only fix test failures that indicate actual code problems
      - If test status is unknown (testing disabled or non-existent):
        * Proceed with caution
        * Make changes that maintain basic test integrity
   
   c) Issue Resolution
      - Understand the problem from issue description and comments
      - Review provided context to identify affected code
      - Plan minimal necessary changes to resolve the issue
      - Consider potential build and test impacts, even if their status is unknown

3. Change Implementation
   - Make precise code modifications only after confirming current file contents
   - Maintain existing patterns and style
   - Preserve important comments and documentation
   - Consider edge cases and error handling
   - Avoid modifying tests that are:
     * Intentionally failing to verify needed fixes
     * Containing bugs (report these instead of changing them)
   - When build/test status is unknown:
     * Make conservative changes
     * Follow standard coding practices
     * Maintain existing error handling patterns

REQUIRED STEPS FOR EVERY EDIT:
1. READ: Use read_file_with_line_numbers to examine the current file state
2. ANALYZE: Confirm the location and content of intended changes
3. PLAN: Document the specific changes needed, categorized by type (build/test/issue)
4. IMPLEMENT: Use edit_file to make the necessary modifications
5. VALIDATE: Ensure changes address the correct problem without breaking other functionality

Never make blind edits without first reading and understanding the current file state.
Never modify tests that are intentionally failing to verify issue fixes.
Report test bugs instead of changing the tests.
When build or test status is unknown, make minimal, conservative changes.
"""

  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {title}
Description: {body}
Comments: {comments}

CONTEXT:
Retrieved context summary:
{summary}

CURRENT STATUS:
{build_status}
{test_status}

Your previous edit (if any):
{patch_info}
"""

  def __init__(self, model: BaseChatModel, root_path: str):
    """Initializes the CodeEditingNode with a language model and root path.

    Sets up the code editing agent with the necessary system prompts, file operation
    tools, and logging configuration.

    Args:
      model: Language model instance that will be used for code analysis and editing.
        Must be a BaseChatModel implementation that supports tool binding.
      root_path: Base directory path for all file operations. All file paths in
        operations will be resolved relative to this root.
    """
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.tools = self._init_tools(root_path)
    self.model_with_tool = model.bind_tools(self.tools)
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.code_editing_node")

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

  def format_issue_comments(self, issue_comments: Sequence[Mapping[str, str]]):
    """Formats issue comments into a readable string.

    Args:
      issue_comments: Sequence of mappings containing username and comment text.

    Returns:
      Formatted string of all comments with usernames.
    """
    formatted_issue_comments = []
    for issue_comment in issue_comments:
      formatted_issue_comments.append(f"{issue_comment['username']}: {issue_comment['comment']}")
    return "\n\n".join(formatted_issue_comments)

  def format_human_message(self, state: IssueAnswerAndFixState) -> HumanMessage:
    """Creates a formatted message for the language model from the current state.

    Args:
      state: Current state containing issue information, build status, and test status.

    Returns:
      HumanMessage instance containing formatted state information.
    """

    # Format build status based on various conditions
    build_status = "Build Status: Unknown (Build check not enabled)"
    if "run_build" in state and state["run_build"]:
      if "exist_build" in state and not state["exist_build"]:
        build_status = "Build Status: Unknown (Build system not found)"
      elif "build_fail_log" in state and state["build_fail_log"]:
        build_status = f"Build Status: FAILING\nBuild Failure Log:\n{state['build_fail_log']}"
      else:
        build_status = "Build Status: Passing"

    # Format test status based on various conditions
    test_status = "Test Status: Unknown (Test check not enabled)"
    if "run_test" in state and state["run_test"]:
      if "exist_test" in state and not state["exist_test"]:
        test_status = "Test Status: Unknown (Test system not found)"
      elif "test_fail_log" in state and state["test_fail_log"]:
        test_status = f"Test Status: FAILING\nTest Failure Log:\n{state['test_fail_log']}"
      else:
        test_status = "Test Status: Passing"

    # Format previous patch information if it exists
    patch_info = ""
    if "patch" in state and state["patch"]:
      patch_info = f"Your previous edit:\n{state['patch']}"

    # Format comments list into a string
    comments_str = self.format_issue_comments(state.get("issue_comments", []))

    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(
        title=state["issue_title"],
        body=state["issue_body"],
        comments=comments_str,
        summary=state["summary"],
        build_status=build_status,
        test_status=test_status,
        patch_info=patch_info,
      )
    )
    return human_message

  def __call__(self, state: IssueAnswerAndFixState):
    """Processes the current state to generate code edits.

    This method takes the current state, formats it into messages for the
    language model, and generates appropriate code modifications based on
    the issue information and status.

    Args:
      state: Current state containing issue information and status.

    Returns:
      Dictionary that will update the state with the model's response messages.
    """
    message_history = [self.system_prompt, self.format_human_message(state)] + state[
      "code_edit_messages"
    ]

    response = self.model_with_tool.invoke(message_history)
    self._logger.debug(f"CodeEditingNode response:\n{response}")
    return {"code_edit_messages": [response]}
