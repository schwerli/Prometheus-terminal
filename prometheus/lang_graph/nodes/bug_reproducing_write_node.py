import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState


class BugReproducingWriteStructuredOutput(BaseModel):
  bug_reproducing_code: str = Field(description="The self-contained code that reproduce the bug")


class BugReproducingWriteNode:
  SYS_PROMPT = """\
You are an agent that writes test cases for reproduce the reported bug.
Your role is to create minimal test code that fails because of the reported bug, and the same test code should pass if the bug is fixed.

REQUIREMENTS:
1. Create minimal test code that uses the EXISTING implementation - do NOT reimplement the buggy code
2. Use proper test assertions (assert statements or testing framework assertions)
3. The test should FAIL if the bug still exists
4. The test should PASS when the bug is fixed
5. Include necessary imports to run the test
6. Add clear comments explaining the expected vs actual behavior
7. If the issue description or comments contain specific examples of failing cases, use those exact examples in your test cases

INPUT EXAMPLE:
```
ISSUE INFORMATION:
Title: JSON parser fails with empty arrays
Description: The JsonParser class crashes when trying to parse an empty array.

Minimal reproducing example:
```python
from src.json.parser import JsonParser

parser = JsonParser()
result = parser.parse_array(['[', ']']) # Raise ValueError!
```
Comments: ["Also fails with nested empty arrays like [[], []]"]

Bug context summary:
### Relevant Context Regarding the Reported Issue with JsonParser
#### Source Code
##### Class: JsonParser
File: `src/json/parser.py`
Lines 10-25:
```python
class JsonParser:
    def parse_array(self, tokens: list) -> list:
        if len(tokens) < 2:  # Need at least [ and ]
            raise ValueError("Invalid array")
        
        if tokens[0] != '[' or tokens[-1] != ']':
            raise ValueError("Array must start with [ and end with ]")
            
        # Extract contents between brackets
        contents = tokens[1:-1]
        
        # Parse contents (bug: doesn't handle empty array case)
        if not contents:
            raise ValueError("Empty array not supported")
            
        return self._parse_array_contents(contents)
```
### Potential Issues
1. The parser explicitly raises an error for empty arrays instead of returning an empty list
2. The error handling for nested empty arrays is missing
```

EXPECTED OUTPUT:
```python
import unittest
from src.json.parser import JsonParser

class TestJsonParser(unittest.TestCase):
    def test_empty_array_parsing(self):
        # Bug: JsonParser fails to parse empty arrays
        # Test will pass when the bug is fixed (empty arrays are handled correctly)
        parser = JsonParser()
        
        # Should return empty list, not raise an exception
        result = parser.parse_array(['[', ']'])
        self.assertEqual(result, [])
        
    def test_nested_empty_arrays(self):
        # Bug: JsonParser fails with nested empty arrays
        # Test will pass when the bug is fixed
        parser = JsonParser()
        
        result = parser.parse_array(['[', '[', ']', ',', '[', ']', ']'])
        self.assertEqual(result, [[], []])

if __name__ == '__main__':
    unittest.main()
```

RESPONSE FORMAT:
1. Import and use the EXISTING implementation
2. Write minimal test code using proper assertions
3. Tests should:
   - FAIL while the bug exists
   - PASS when the bug is fixed
4. Use appropriate testing framework (unittest, pytest, etc.)

Remember: Write tests that verify the correct behavior, not the buggy behavior!
""".replace("{", "{{").replace("}", "}}")

  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {title}
Description: {body}
Comments: {comments}

Bug context summary:
{bug_context}

Previous bug reproducing code
{previous_bug_reproducing_code}

Previous bug reproducing fail log
{previous_bug_reproducing_fail_log}
"""

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{issue_info}")]
    )
    structured_llm = model.with_structured_output(BugReproducingWriteStructuredOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_reproducing_write_node")

  def format_human_message(self, state: BugReproductionState):
    previous_bug_reproducing_code = ""
    if "bug_reproducing_code" in state and state["bug_reproducing_code"]:
      previous_bug_reproducing_code = state["bug_reproducing_code"]
    previous_bug_reproducing_fail_log = ""
    if "reproduced_bug_failure_log" in state and state["reproduced_bug_failure_log"]:
      previous_bug_reproducing_fail_log = state["reproduced_bug_failure_log"]
    return self.HUMAN_PROMPT.format(
      title=state["issue_title"],
      body=state["issue_body"],
      comments=state["issue_comments"],
      bug_context=state["bug_context"],
      previous_bug_reproducing_code=previous_bug_reproducing_code,
      previous_bug_reproducing_fail_log=previous_bug_reproducing_fail_log,
    )

  def __call__(self, state: BugReproductionState):
    issue_info = self.format_human_message(state)
    response = self.model.invoke({"issue_info": issue_info})

    self._logger.debug(f"BugReproducingWriteNode response:\n{response}")
    return {"bug_reproducing_code": response.bug_reproducing_code}
