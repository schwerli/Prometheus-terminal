import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.tools import file_operation


class BugReproducingWriteNode:
  SYS_PROMPT = """\
You are an agent that writes or modifies TEST FILES ONLY to reproduce reported bugs. Your primary goal is
to create or update test cases that reliably demonstrate bug behavior described in the issue.

STRICT FILE MODIFICATION RULES:
1. If previous_reproducing_file is provided:
   - You may ONLY modify that specific file
   - You are STRICTLY FORBIDDEN from touching ANY other files
   - If the file needs changes, use edit_file ONLY on previous_reproducing_file

2. If NO previous_reproducing_file is provided:
   - You must create a NEW test file using create_file
   - You are STRICTLY FORBIDDEN from modifying ANY existing files
   - Never append to or modify existing test files

ABSOLUTELY FORBIDDEN ACTIONS:
- NEVER modify any source code files
- NEVER modify any existing test files except previous_reproducing_file
- NEVER append tests to existing files, even if they seem relevant
- NEVER suggest modifications to files you can't edit

Test Implementation:
1. New Test Files:
   - Create ONLY when no "Previous repreducing file" exists
   - Place in appropriate test directory
   - Follow naming conventions (test_*.py, *_test.py)

2. Previous File Modification:
   - Edit ONLY if it's explicitly provided as "Previous repreducing file"
   - Keep existing structure and patterns
   - Only use edit_file on this specific file

3. Test Requirements:
   - Include necessary imports and fixtures
   - Document reproduction steps
   - Match reported bug behavior
   - Ensure deterministic results

Remember: You have permission to modify ONLY the "Previous repreducing file" OR create a new file. ANY other file modifications are strictly forbidden.
"""

  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {title}
Description: {body}
Comments: {comments}

Project structure:
{project_structure}

Bug context summary:
{bug_context}

Previous repreducing file:
{previous_reproducing_file}

Previous repreducing attempt:
{previous_reproducing_attempt}
"""

  def __init__(self, model: BaseChatModel, kg: KnowledgeGraph):
    self.kg = kg
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.tools = self._init_tools(kg.get_local_path())
    self.model_with_tool = model.bind_tools(self.tools)
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_reproducing_write_node")

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

    edit_file_fn = functools.partial(file_operation.edit_file, root_path=root_path)
    edit_file_tool = StructuredTool.from_function(
      func=edit_file_fn,
      name=file_operation.edit_file.__name__,
      description=file_operation.EDIT_FILE_DESCRIPTION,
      args_schema=file_operation.EditFileInput,
    )
    tools.append(edit_file_tool)

    return tools

  def format_human_message(self, state: BugReproductionState):
    previous_reproducing_file = ""
    if "reproduced_bug_file" in state and state["reproduced_bug_file"]:
      previous_reproducing_file = state["reproduced_bug_file"]

    previous_reproducing_attempt = ""
    if (
      "last_bug_reproducing_execute_message" in state
      and state["last_bug_reproducing_execute_message"]
    ):
      previous_reproducing_attempt = state["last_bug_reproducing_execute_message"].content

    return HumanMessage(
      self.HUMAN_PROMPT.format(
        title=state["issue_title"],
        body=state["issue_body"],
        comments=state["issue_comments"],
        project_structure=self.kg.get_file_tree(),
        bug_context=state["bug_context"],
        previous_reproducing_file=previous_reproducing_file,
        previous_reproducing_attempt=previous_reproducing_attempt,
      )
    )

  def __call__(self, state: BugReproductionState):
    message_history = [self.system_prompt, self.format_human_message(state)] + state[
      "bug_reproducing_write_messages"
    ]

    response = self.model_with_tool.invoke(message_history)
    self._logger.debug(f"BugReproducingWriteNode response:\n{response}")
    return {"bug_reproducing_write_messages": [response]}
