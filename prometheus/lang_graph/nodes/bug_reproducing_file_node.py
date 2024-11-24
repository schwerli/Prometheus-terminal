import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.tools import file_operation


class BugReproducingFileNode:
  SYS_PROMPT = """\
You are a specialized file management agent responsible for handling bug reproduction code. Your task is to manage the placement and creation of bug reproducing code files in the project.

THOUGHT PROCESS:
1. Analyze the input:
   - Check if a bug reproducing file path is provided
   - Review the code content to determine appropriate file type
   - Examine project structure for logical file placement
2. Plan file operations:
   - Determine if existing file needs deletion
   - Choose appropriate file location and name
   - Plan verification steps
3. Execute operations:
   - Perform necessary file operations in correct order
   - Verify results
   - Prepare response message

REQUIREMENTS:
1. Handle existing files:
   - Delete file if it exists at the provided path
   - Create new file at the same path
   - Write provided code into the file

2. Handle new files:
   - Choose appropriate location (typically in 'tests' directory)
   - Create descriptive filename based on bug
   - Write provided code into the file

3. File naming and placement:
   - Use correct file extension for code language
   - Place files in logical project locations
   - Use clear, descriptive names reflecting the bug

4. Verification:
   - Confirm all file operations succeed
   - Only modify intended files
   - Verify final file contents

AVAILABLE TOOLS:
- read_file: Verify file existence and contents
- create_file: Write new files with provided content
- delete: Remove existing files

<example>
<input>
<bug_reproducing_code>
import unittest
from myproject.parser import JsonParser

class TestEmptyArrays(unittest.TestCase):
    def test_empty_array(self):
        parser = JsonParser()
        result = parser.parse_array(['[', ']'])
        self.assertEqual(result, [])
</bug_reproducing_code>
<reproduced_bug_file>tests/parser/test_empty_arrays.py</reproduced_bug_file>
</input>

<reasoning>
1. Input Analysis:
   - Existing file path provided: tests/parser/test_empty_arrays.py
   - Content is Python unittest code
   - Tests JSON parser empty array bug

2. Operation Planning:
   - Need to delete existing file first
   - Will create new file at same path
   - Should verify file contents after creation

3. Execution Steps:
   - Delete existing file
   - Create new file with provided code
   - Verify content matches
</reasoning>

<tool_usage>
1. Delete existing file:
   delete(path="tests/parser/test_empty_arrays.py")

2. Create new file:
   create_file(
     path="tests/parser/test_empty_arrays.py",
     content=provided_code
   )

3. Verify content:
   content = read_file(path="tests/parser/test_empty_arrays.py")
   # Verify content matches provided code
</tool_usage>

<output>
Bug reproducing code has been written to: tests/parser/test_empty_arrays.py
</output>
</example>

RESPONSE FORMAT:
Your response should follow this structure:
<thought_process>
1. Analyze the provided code and path
2. Plan required file operations
3. List verification steps
</thought_process>

<operations>
Execute necessary file operations using available tools
</operations>

<response>
"Bug reproducing code has been written to: {file_path}"
</response>

Remember:
- Only modify the specific target file
- Always verify operations succeed
- Provide exactly the required response message
- Use tools in the correct order (delete -> create -> verify)
"""

  HUMAN_PROMPT = """\
Bug reproducing code:
{bug_reproducing_code}

Bug reproducing file:
{reproduced_bug_file}
"""

  def __init__(
    self,
    model: BaseChatModel,
    kg: KnowledgeGraph,
  ):
    self.kg = kg
    self.tools = self._init_tools(kg.get_local_path())
    self.model_with_tools = model.bind_tools(self.tools)
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_reproducing_file_node")

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

    return tools

  def format_human_message(self, state: BugReproductionState) -> HumanMessage:
    reproduced_bug_file = ""
    if "reproduced_bug_file" in state and state["reproduced_bug_file"]:
      reproduced_bug_file = state["reproduced_bug_file"]
    return HumanMessage(
      self.HUMAN_PROMPT.format(
        bug_reproducing_code=state["bug_reproducing_code"],
        reproduced_bug_file=reproduced_bug_file,
        project_structure=self.kg.get_file_tree(),
      )
    )

  def __call__(self, state: BugReproductionState):
    message_history = [self.system_prompt, self.format_human_message(state)] + state[
      "bug_reproducing_file_messages"
    ]

    response = self.model_with_tools.invoke(message_history)
    self._logger.debug(f"BugReproducingFileNode response:\n{response}")
    return {"bug_reproducing_file_messages": [response]}