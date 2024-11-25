import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.utils.issue_util import format_agent_tool_message_history, format_issue_comments


class BugReproducingStructuredOutput(BaseModel):
  reproduced_bug: bool = Field(
    description="True ONLY if test fails as described in the issue and uses provided examples if any exist"
  )
  reproduced_bug_failure_log: str = Field(
    description="Explanation of why the test didn't properly reproduce the bug (if test passes, has different error, doesn't use issue examples, etc). Empty if bug was reproduced correctly"
  )
  reproduced_bug_file: str = Field(
    description="The relative path of the file that reproduces the bug"
  )
  reproduced_bug_commands: Sequence[str] = Field(
    description="A list of commands run to the single file to reproduce the bug"
  )


class BugReproducingStructuredNode:
  SYS_PROMPT = """\
You are an agent that verifies if a test properly reproduces a reported bug. You analyze:
1. Issue description and any provided examples
2. The test code written to reproduce the bug
3. The test execution results

For reproduced_bug to be True, the test MUST:
1. Fail (since bug isn't fixed yet)
2. Fail in exactly the same way described in the issue
3. Use the exact examples from the issue if any were provided
4. Demonstrate the same underlying problem

Set reproduced_bug_failure_log when the test:
- Passes (explain that passing means bug isn't detected)
- Fails differently than described (explain how error differs)
- Doesn't use provided examples (explain what example should be used)
- Tests wrong behavior (explain what behavior should be tested)
- Has any other issues (explain the problem)

Example 1 - Correct Reproduction:
```
Issue:
Title: Array.pop() crashes on empty array
Description: Calling pop() on empty array throws "Cannot read property 'length' of undefined" but should throw "Array is empty"
Example: let arr = []; arr.pop(); // Shows wrong error

Test:
def test_empty_array_pop():
    arr = []
    with pytest.raises(ValueError, match="Cannot read property 'length' of undefined"):
        arr.pop()

Result: Test failed with "Cannot read property 'length' of undefined"

Output:
{
    "reproduced_bug": true,
    "reproduced_bug_failure_log": "",
    "reproduced_bug_file": "tests/test_array.py",
    "reproduced_bug_commands": ["pytest tests/test_array.py"]
}
```

Example 2 - Wrong Error:
```
Issue:
Title: Array.pop() crashes on empty array
Description: Calling pop() on empty array throws "Cannot read property 'length' of undefined" but should throw "Array is empty"
Example: let arr = []; arr.pop(); // Shows wrong error

Test:
def test_empty_array_pop():
    arr = []
    with pytest.raises(IndexError):
        arr.pop()

Result: Test failed with "IndexError: pop from empty list"

Output:
{
    "reproduced_bug": false,
    "reproduced_bug_failure_log": "Test fails with IndexError but issue describes 'Cannot read property length' error. Test needs to verify the specific error message reported in the bug.",
    "reproduced_bug_file": "tests/test_array.py",
    "reproduced_bug_commands": ["pytest tests/test_array.py"]
}
```

Example 3 - Wrong Example:
```
Issue:
Title: Array.pop() crashes on empty array
Description: Calling pop() on empty array throws "Cannot read property 'length' of undefined" but should throw "Array is empty"
Example: let arr = []; arr.pop(); // Shows wrong error

Test:
def test_array_pop():
    arr = [1, 2, 3]
    arr.pop()
    assert len(arr) == 2

Result: Test passed

Output:
{
    "reproduced_bug": false,
    "reproduced_bug_failure_log": "Test passes and doesn't use the empty array example from the issue. Test should verify pop() behavior on an empty array as shown in the example.",
    "reproduced_bug_file": "tests/test_array.py", 
    "reproduced_bug_commands": ["pytest tests/test_array.py"]
}
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
      bug_reproducing_code=state["bug_reproducing_write_messages"][-1].content,
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
