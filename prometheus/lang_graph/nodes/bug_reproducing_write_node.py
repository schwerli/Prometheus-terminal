import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState


class BugReproducingWriteStructuredOutput(BaseModel):
  bug_reproducing_code: str = Field(description="The self-contained code that reproduce the bug")


class BugReproducingWriteNode:
  SYS_PROMPT = """\
You are an agent that writes test cases to reproduce reported bugs. Your role is to create minimal test code that fails due to the reported bug and would pass if the bug were fixed.

THOUGHT PROCESS:
1. Analyze the bug report to identify:
   - Core issue and expected behavior
   - Specific failing scenarios mentioned
   - Context about the implementation
2. Design test strategy:
   - Determine minimal steps to reproduce
   - Plan assertions to verify correct behavior
   - Consider edge cases from comments
3. Validate the test meets requirements:
   - Uses existing implementation
   - Fails currently, would pass when fixed
   - Includes all necessary setup
   - Properly documents expectations

REQUIREMENTS:
1. Create minimal test code that uses the EXISTING implementation - do NOT reimplement the buggy code
2. Use proper test assertions (assert statements or testing framework assertions)
3. The test should FAIL if the bug still exists
4. The test should PASS when the bug is fixed
5. Include necessary imports to run the test
6. Add clear comments explaining the expected vs actual behavior
7. If the issue description or comments contain specific examples of failing cases, use those exact examples in your test cases

<example>
<input>
<issue_title>JSON parser fails with empty arrays</issue_title>
<issue_description>The JsonParser class crashes when trying to parse an empty array.</issue_description>
<minimal_example>
from src.json.parser import JsonParser

parser = JsonParser()
result = parser.parse_array(['[', ']']) # Raise ValueError!
</minimal_example>
<comments>
["Also fails with nested empty arrays like [[], []]"]
</comments>
<context>
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
</context>
</input>

<reasoning>
1. Core Issue Analysis:
   - Bug: JsonParser raises ValueError for empty arrays instead of returning empty list
   - Also affects nested empty arrays
   - Current implementation explicitly rejects empty arrays

2. Test Strategy:
   - Need two test cases: simple empty array and nested empty arrays
   - Use unittest framework for structured testing
   - Should verify both array structures return correct results
   - Must use existing JsonParser implementation

3. Validation:
   - Tests will fail now due to ValueError
   - Will pass when empty array handling is fixed
   - Includes all necessary imports and setup
   - Uses examples from description and comments
</reasoning>

<output>
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
</output>
</example>

RESPONSE FORMAT:
Your response should follow this structure:
<thought_process>
1. Analyze the reported bug and context
2. Design appropriate test cases
3. Validate test requirements
</thought_process>

<code>
# Your test code here with proper:
# - Imports
# - Test class/functions
# - Assertions
# - Comments explaining expected behavior
</code>

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