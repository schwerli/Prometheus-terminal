import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.utils.issue_util import format_agent_tool_message_history, format_issue_comments


class BugReproducingStructuredOutput(BaseModel):
  reproduced_bug: bool = Field(
    description="True if test fails in a way that demonstrates the underlying bug"
  )
  reproduced_bug_failure_log: str = Field(
    description="If test passes or fails in a way unrelated to the bug, explain why. Empty if failure demonstrates the bug"
  )
  reproduced_bug_file: str = Field(
    description="The relative path of the file that reproduces the bug"
  )
  reproduced_bug_commands: Sequence[str] = Field(
    description="A list of commands run to the single file to reproduce the bug"
  )


class BugReproducingStructuredNode:
  SYS_PROMPT = """\
You are an agent that analyzes test results during the initial bug verification phase, before any fixes are attempted.
You'll receive:
1. The GitHub issue describing the bug (title, description, comments)
2. The bug reproducing code written by another agent
3. The location where the test file was written
4. The execution results of the test

Your task is to verify if the test properly demonstrates the underlying bug:
- Test MUST FAIL at this stage (we haven't fixed the bug yet)
- The failure should demonstrate the same underlying issue described in the bug report
- Exact error messages or failure types may vary, as long as they expose the same core problem
- Test passing means it's not properly detecting the bug

Provide the following information:
1. reproduced_bug (boolean): True if test fails in a way that demonstrates the underlying bug, False if it passes or fails unrelated to the bug
2. reproduced_bug_failure_log (string): Empty if failure demonstrates the bug. Otherwise explain why the test isn't properly exposing the reported issue
3. reproduced_bug_file (string): The exact relative path to the test file
4. reproduced_bug_commands (list of strings): Commands needed to run the test

Example of Correct Test (Shows Bug):
```
ISSUE INFORMATION:
Title: JSON parser fails with empty arrays
Description: The JsonParser class crashes when trying to parse an empty array "[]". This should be valid JSON!
Comments: ["Also fails with nested empty arrays like [[], []]"]

Bug reproduction code:
def test_empty_array_parsing():
    parser = JsonParser()
    assert parser.parse_array(['[', ']']) == []

Test file message:
Test has been written to: tests/test_json_parser.py

Test Execute Messages:
Tool Calls:
run_command: pytest tests/test_json_parser.py

Output:
============================= test session starts ==============================
platform linux -- Python 3.9.20, pytest-7.4.4, pluggy-1.0.0
collected 1 item

tests/test_json_parser.py F                                              [100%]

================================= FAILURES ==================================
_________________________ test_empty_array_parsing _________________________
    def test_empty_array_parsing():
>       assert parser.parse_array(['[', ']']) == []
E       AssertionError: assert None == []

tests/test_json_parser.py:7: AssertionError
============================== 1 failed in 0.16s =============================
```

Example Output for Correct Test:
```python
{
    "reproduced_bug": True,  # True because test shows empty arrays aren't handled properly
    "reproduced_bug_failure_log": "",  # Empty because failure demonstrates the bug (empty arrays not working)
    "reproduced_bug_file": "tests/test_json_parser.py",
    "reproduced_bug_commands": ["pytest tests/test_json_parser.py"]
}
```

Example of Issue With Test:
```
ISSUE INFORMATION:
Title: JSON parser fails with empty arrays
Description: The JsonParser class crashes when trying to parse an empty array "[]". This should be valid JSON!
Comments: ["Also fails with nested empty arrays like [[], []]"]

Bug reproduction code:
def test_array_parsing():
    parser = JsonParser()
    assert parser.parse_array(['[', '1', ']']) == [1]

Test file message:
Test has been written to: tests/test_json_parser.py

Test Execute Messages:
Tool Calls:
run_command: pytest tests/test_json_parser.py

Output:
============================= test session starts ==============================
platform linux -- Python 3.9.20, pytest-7.4.4, pluggy-1.0.0
collected 1 item

tests/test_json_parser.py F                                              [100%]

================================= FAILURES ==================================
_________________________ test_array_parsing _________________________
    def test_array_parsing():
>       assert parser.parse_array(['[', '1', ']']) == [1]
E       TypeError: Expected string but got int

tests/test_json_parser.py:7: TypeError
============================== 1 failed in 0.16s =============================
```

Example Output for Issue With Test:
```python
{
    "reproduced_bug": False,  # False because test is checking non-empty array parsing instead of empty array handling
    "reproduced_bug_failure_log": "Test is failing due to type conversion with non-empty arrays, not the reported empty array handling issue. The test needs to check empty array parsing to verify the reported bug.",
    "reproduced_bug_file": "tests/test_json_parser.py",
    "reproduced_bug_commands": ["pytest tests/test_json_parser.py"]
}
```

Remember:
- Test MUST fail at this stage since we haven't fixed the bug
- Focus on whether the failure demonstrates the core issue from the bug report
- The exact nature of the failure (error type, message) is less important than showing the underlying problem
- A passing test means the bug isn't being checked
- Path separators should match the format used in the input
- Commands should be properly formatted and executable
""".replace("{", "{{").replace("}", "}}")

  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {title}
Description: {body}
Comments: {comments}

Bug reproducing file message:
{bug_reproducing_file_message}

Bug reproduction code:
{bug_reproducing_code}

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
      title=state["issue_title"],
      body=state["issue_body"],
      comments=format_issue_comments(state["issue_comments"]),
      bug_reproducing_code=state["bug_reproducing_code"],
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
