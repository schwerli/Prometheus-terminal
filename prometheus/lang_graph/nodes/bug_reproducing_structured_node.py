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
You are an agent that analyzes bug reproduction results and extracts structured information. You'll receive two pieces of information:
1. The location where the bug reproducing file was written
2. The execution results of the bug reproduction attempt

Your task is to verify if the test correctly demonstrates the bug by checking for the exact output message:
- "Bug reproduced" indicates a successful reproduction of the reported bug
- If this message is not present, the reproduction attempt has failed

Provide the following information:
1. reproduced_bug (boolean): True if the test output contains exactly "Bug reproduced", False otherwise
2. reproduced_bug_failure_log (string): If the message "Bug reproduced" is not found, explain why the reproduction failed. Empty string if successful
3. reproduced_bug_file (string): The exact relative path to the test file from the file operations
4. reproduced_bug_commands (list of strings): Commands needed to run the test, executable as-is

Example of Successful Reproduction:
```
# Bug reproducing file message:
Bug reproducing code has been written to: tests/test_invalid_url.py

# Bug Reproducing Execute Messages:
Tool Calls:
run_command: pytest tests/test_invalid_url.py

Output:
============================= test session starts ==============================
platform linux -- Python 3.9.20, pytest-7.4.4, pluggy-1.0.0
collected 1 item

tests/test_invalid_url.py .                                              [100%]

Bug reproduced
============================== 1 passed in 0.16s ==============================
```

Example Output for Success:
```python
{
    "reproduced_bug": True,  # True because the test outputs "Bug reproduced"
    "reproduced_bug_failure_log": "",
    "reproduced_bug_file": "tests/test_invalid_url.py",
    "reproduced_bug_commands": ["pytest tests/test_invalid_url.py"]
}
```

Example of Failed Reproduction:
```
# Bug reproducing file message:
Bug reproducing code has been written to: tests/test_parsing.py

# Bug Reproducing Execute Messages:
Tool Calls:
run_command: pytest tests/test_parsing.py

Output:
============================= test session starts ==============================
platform linux -- Python 3.9.20, pytest-7.4.4, pluggy-1.0.0
collected 1 item

tests/test_parsing.py F                                                  [100%]

================================= FAILURES ==================================
______________________________ test_parsing _______________________________
    def test_parsing():
>       result = parse_config("invalid/path")
E       FileNotFoundError: [Errno 2] No such file or directory: 'invalid/path'

tests/test_parsing.py:7: FileNotFoundError
=========================== short test summary info ==========================
FAILED tests/test_parsing.py::test_parsing - FileNotFoundError: [Errno 2] ...
============================== 1 failed in 0.16s =============================
```

Example Output for Failure:
```python
{
    "reproduced_bug": False,  # False because test failed with exception instead of outputting "Bug reproduced"
    "reproduced_bug_failure_log": "Test failed with FileNotFoundError instead of demonstrating the bug. The test should handle file paths properly and output 'Bug reproduced' when showing the issue.",
    "reproduced_bug_file": "tests/test_parsing.py",
    "reproduced_bug_commands": ["pytest tests/test_parsing.py"]
}
```

Remember:
- Always provide all four fields
- Focus on whether the test outputs exactly "Bug reproduced"
- Keep failure logs focused on substantive problems, not message differences
- Ensure commands are properly formatted and executable
- Path separators should match the format used in the input
""".replace("{", "{{").replace("}", "}}")

  HUMAN_PROMPT = """\
Bug reproducing file message:
{bug_reproducing_file_message}

Log from executing bug reproducing file:
{bug_reproducing_log}
"""

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{bug_reproducing_info}")]
    )
    structured_llm = model.with_structured_output(BugReproducingStructuredOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_reproducing_structured_node")

  def __call__(self, state: BugReproductionState):
    bug_reproducing_log = format_agent_tool_message_history(
      state["bug_reproducing_execute_messages"]
    )
    bug_reproducing_info = self.HUMAN_PROMPT.format(
      bug_reproducing_file_message=state["bug_reproducing_file_messages"][-1].content,
      bug_reproducing_log=bug_reproducing_log,
    )

    response = self.model.invoke({"bug_reproducing_info": bug_reproducing_info})
    self._logger.debug(f"BugReproducingStructuredNode response:\n{response}")

    return {
      "reproduced_bug": response.reproduced_bug,
      "reproduced_bug_failure_log": response.reproduced_bug_failure_log,
      "reproduced_bug_file": response.reproduced_bug_file,
      "reproduced_bug_commands": response.reproduced_bug_commands,
    }
