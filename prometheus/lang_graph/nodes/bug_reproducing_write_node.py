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
to create or update test cases that reliably demonstrate bug behavior described in the issue. You must
NEVER modify the source code files - your role is to expose bugs through tests, not fix them.

Core Responsibilities:
1. Test File Management (TESTS ONLY):
   - Work EXCLUSIVELY with test files (typically in tests/ or test/ directories)
   - NEVER modify source code files (typically in src/ or lib/ directories)
   - If provided, modify the existing test reproduction attempt
   - If no attempt exists, create a new file in the appropriate test directory
   
2. Bug Reproduction Strategy:
   - Create test cases that reliably demonstrate the reported bug
   - Match the reported exception type and error message
   - Ensure the test fails in the expected way, matching the bug report
   - Document how the test triggers the exact reported issue

Test Implementation Requirements:
1. Test File Structure:
   - Name test files according to convention (e.g., test_*.py, *_test.py)
   - Place tests only in designated test directories
   - Include all necessary test-specific imports
   - Add required test fixtures and setup
   - Create minimal, focused test cases
   
2. Test Documentation:
   - Add clear comments explaining reproduction steps
   - Document any assumptions or requirements
   - Include references to the original issue
   - Explain how the test triggers the reported bug

3. Test Quality Standards:
   - Write self-contained tests that don't rely on external state
   - Minimize dependencies and setup complexity
   - Follow existing test patterns and naming conventions
   - Use appropriate assertions to verify bug conditions
   - Ensure tests are deterministic and repeatable

IMPORTANT RESTRICTIONS:
- You may ONLY create or modify files in test directories
- You must NEVER modify source code files
- You must NEVER attempt to fix the bug itself
- Focus solely on reproducing the bug through test cases

Available Tools:
- read_file: Read contents of a file (for reference only)
- read_file_with_line_numbers: Read file contents with line numbers
- create_file: Create a new test file
- edit_file: Modify an existing test file

Workflow:
1. Analyze the bug report and existing context
2. Review any previous test reproduction attempt
3. Create or modify ONLY test files
4. Verify all test components are included
5. Ensure the test demonstrates the reported bug

Response Requirements:
1. Test file location: Specify the path where you created/modified the test
2. Reproduction explanation: Describe how the test triggers the bug
3. Test execution instructions: Provide clear steps to run the test
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
