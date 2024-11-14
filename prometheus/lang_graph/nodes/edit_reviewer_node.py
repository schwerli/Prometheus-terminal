import functools
import logging
from typing import Mapping, Sequence

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from prometheus.tools import file_operation


class EditReviewerNode:
  SYS_PROMPT = """\
You are a code review assistant. Your task is to thoroughly review patches generated to fix GitHub issues. For each review, you will receive:
1. The original issue information (title, description, and comments)
2. Retrieved context about relevant code
3. The generated patch

You have access to file reading tools - you can use them to examine any files mentioned in the context or patch if you need to verify existing patterns, test coverage, or implementation details.

Focus your review on these key aspects:

FUNCTIONALITY
- Does the patch properly address the issue described?
- Are there any edge cases or scenarios not handled?
- Could this fix introduce new problems?

CODE QUALITY
- Does the change follow the patterns shown in the retrieved context?
- Are there any critical issues with the implementation (e.g., missing arguments, incorrect error handling, potential race conditions)?
- IGNORE minor nits (e.g., variable naming) unless they impact functionality

SCOPE
- Are the changes appropriately scoped to fix the issue?
- If the patch modifies files not mentioned in the issue/context, evaluate if these changes are necessary (e.g., for fixing build/test issues)
- Flag if changes seem unnecessarily broad or touch unrelated areas

RESPONSE FORMAT:
1. Start with a clear "VERDICT" section: APPROVE or REQUEST_CHANGES
2. Provide a "SUMMARY" section with a brief overview of your assessment
3. If issues are found, list them in an "ISSUES" section, ordered by severity
4. (Optional) Include a "NOTES" section for any additional observations

Example response:
VERDICT: REQUEST_CHANGES

SUMMARY:
The patch addresses the core functionality but has a critical issue with error handling that could cause crashes in production.

ISSUES:
1. [Critical] The new error handler on line 45 swallows exceptions silently
2. [Important] The database connection isn't properly closed in the error path

Remember: Focus on substantive issues that affect functionality, reliability, or maintainability. Don't get caught up in minor nits.
"""

  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {title}
Description: {body}
Comments: {comments}

CONTEXT:
Retrieved context summary:
{summary}

GENERATED PATCH:
{patch}
"""

  def __init__(self, model: BaseChatModel, root_path: str):
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
    comments_str = self.format_issue_comments(state.get("issue_comments", []))

    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(
        title=state["issue_title"],
        body=state["issue_body"],
        comments=comments_str,
        summary=state["summary"],
        patch=state["patch"],
      )
    )
    return human_message

  def __call__(self, state: IssueAnswerAndFixState):
    """Review the patch generated by CodeEditingNode.

    This method takes the current state, formats it into messages for the
    language model, and reviews the generated patch.

    Args:
      state: Current state containing issue information and status.

    Returns:
      Dictionary that will update the state with the model's response messages.
    """
    message_history = [self.system_prompt, self.format_human_message(state)] + state[
      "edit_reviewer_messages"
    ]

    response = self.model_with_tool.invoke(message_history)
    self._logger.debug(f"EditReviewerNode response:\n{response}")
    return {"edit_reviewer_messages": [response]}
