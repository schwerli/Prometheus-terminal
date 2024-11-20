import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.utils.issue_util import format_agent_tool_message_history


class BugReproducingStructuredOutput(BaseModel):
  reproduced_bug: bool = Field(description="If the written code reproduces the bug")
  reproduced_bug_failure_log: str = Field(
    description="How the bug preduction failed to reproduce the bug, empty string if successful"
  )
  reproduced_bug_file: str = Field(
    description="The relative path of the file that reproduces the bug"
  )
  reproduced_bug_commands: Sequence[str] = Field(
    description="A list of commands run to the single file to reproduce the bug"
  )


class BugReproducingStructuredNode:
  SYS_PROMPT = """\
You are an agent that analyzes the output from a bug reproduction attempt and provides structured information about the results.
Your task is to extract key details and present them in a specific format.

You will receive the raw output from two sources concatenated together:
1. The bug reproducing write messages - showing file operations and test creation
2. The bug reproducing execute messages - showing test execution results

The output includes:
1. The file operations performed (create/edit/read)
2. The test file created or modified
3. Any execution results or error messages
4. Instructions for running the test

Analyze this information and provide the following structured output:

1. reproduced_bug (boolean):
   - True if the test file successfully demonstrates the bug
   - False if the reproduction attempt failed or is incomplete

2. reproduced_bug_failure_log (string):
   - If reproduced_bug is False: Describe why the reproduction failed
   - If reproduced_bug is True: Return an empty string

3. reproduced_bug_file (string):
   - The relative path to the test file that was created or modified
   - Must be extracted from file operations in the output
   - Should match the exact path used in create_file or edit_file operations

4. reproduced_bug_commands (list of strings):
   - List of commands needed to run the reproduction test
   - Include only commands that directly execute the test file
   - Each command should be a separate string
   - Commands should be executable as-is
   - Do not include setup or installation commands

Rules for Processing:
1. Look for explicit file paths in create_file or edit_file operations
2. Extract execution commands from the agent's instructions
3. Determine reproduction success based on whether the agent completed the task
4. If multiple files are mentioned, use only the final test file path
5. Commands must be specific to running the test file, not general setup

Example Input:
```
# Bug Reproducing Write Messages:
Assistant: I've created a test file to reproduce the described bug.

Tool Calls:
create_file:
  path: tests/unit/test_parser_bug.py
  content: [test file content...]

I've created a test file that reproduces the parser bug. Here's how to run it:
1. Navigate to the project root
2. Run: python -m pytest tests/unit/test_parser_bug.py -v

The test should fail with the same IndexError as reported in the bug.

# Bug Reproducing Execute Messages:
Assistant: I'll execute the test file to verify it reproduces the bug.

Tool Calls:
execute_command: python -m pytest tests/unit/test_parser_bug.py -v

Output:
============================= test session starts ==============================
collecting ... collected 1 item

tests/unit/test_parser_bug.py::TestParser::test_invalid_input_raises_index_error FAILED

============================== FAILURES ======================================
TestParser::test_invalid_input_raises_index_error
    def test_invalid_input_raises_index_error():
>       result = parser.parse("invalid input")
E       IndexError: string index out of range

Expected failure successfully reproduced.
```

Example Output:
```python
{
    "reproduced_bug": True,
    "reproduced_bug_failure_log": "",
    "reproduced_bug_file": "tests/unit/test_parser_bug.py",
    "reproduced_bug_commands": ["python -m pytest tests/unit/test_parser_bug.py -v"]
}
```

Remember:
- Always provide all four fields
- Keep failure logs concise but informative
- Ensure commands are properly formatted and executable
- Path separators should match the format used in the input
""".replace("{", "{{").replace("}", "}}")

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{bug_reproducing_attempt_messages}")]
    )
    structured_llm = model.with_structured_output(BugReproducingStructuredOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_reproducing_structured_node")

  def __call__(self, state: BugReproductionState):
    bug_reproducing_write_messages = format_agent_tool_message_history(
      state["bug_reproducing_write_messages"]
    )

    bug_reproducing_execute_messages = format_agent_tool_message_history(
      state["bug_reproducing_execute_messages"]
    )

    bug_reproducing_attempt_messages = (
      bug_reproducing_write_messages + "\n\n" + bug_reproducing_execute_messages
    )
    response = self.model.invoke(
      {"bug_reproducing_attempt_messages": bug_reproducing_attempt_messages}
    )
    self._logger.debug(f"BugReproducingStructuredNode response:\n{response}")

    return {
      "reproduced_bug": response.reproduced_bug,
      "reproduced_bug_failure_log": response.reproduced_bug_failure_log,
      "reproduced_bug_file": response.reproduced_bug_file,
      "reproduced_bug_commands": response.reproduced_bug_commands,
    }
