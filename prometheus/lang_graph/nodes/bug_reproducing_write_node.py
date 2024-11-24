import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.tools import file_operation


class BugReproducingWriteNode:
  SYS_PROMPT = '''\
Write a minimal, self-contained test case that reproduces the reported bug. The test should:
- Be completely self-contained with all necessary imports
- Use the existing implementation (don't modify the buggy code)
- Include proper assertions that will fail now and pass when fixed
- Focus on the core issue without additional test cases

Output only the test code without explanations or commentary.

When writing test cases, follow this thought process:

1. Understand the Bug
   - What is the core issue?
   - What should happen vs what actually happens?
   - Are there specific examples in the bug report?

2. Analyze Available Context
   - Review the code structure from bug context
   - Check previous test attempts and failure logs
   - Identify relevant classes and methods

3. Design Test Strategy
   - Plan minimal steps to reproduce
   - Choose appropriate assertions
   - Consider edge cases from comments
   - Use the existing implementation

4. Implement and Validate
   - Write self-contained test
   - Include all necessary imports
   - Add clear documentation
   - Verify test meets requirements

Key Requirements:
- Test must be minimal and self-contained
- Use the EXISTING implementation (don't rewrite the buggy code)
- Include proper assertions
- Test should fail now and pass when fixed
- Document expected vs actual behavior
- Prioritize examples from the bug report

<example>
<bug_report>
Title: JSON parser fails with empty arrays
Description: The JsonParser class crashes when trying to parse an empty array - it should return an empty list but instead raises a ValueError.
Example:
```python
parser = JsonParser()
result = parser.parse_array(['[', ']'])  # Raises ValueError!
```
Comments: Also fails with nested empty arrays like [[], []]. This is blocking our ability to parse valid JSON containing empty arrays.
</bug_report>

<context>
### JSON Parser Implementation
**File**: json/parser.py
```python
class JsonParser:
    def parse_array(self, tokens: list) -> list:
        """Parse a JSON array from a list of tokens.
        
        Args:
            tokens: List of JSON tokens
            
        Returns:
            Parsed array as Python list
            
        Raises:
            ValueError: If array format is invalid
        """
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
    
    def _parse_array_contents(self, contents: list) -> list:
        """Parse the contents of a JSON array.
        
        Args:
            contents: Tokens between [ and ]
            
        Returns:
            List of parsed values
        """
        result = []
        current_item = []
        
        for token in contents:
            if token == ',':
                if current_item:
                    result.append(self._parse_value(current_item))
                    current_item = []
            else:
                current_item.append(token)
                
        if current_item:
            result.append(self._parse_value(current_item))
            
        return result

### Existing Tests
**File**: tests/test_json_parser.py
```python
import unittest
from json.parser import JsonParser

class TestJsonParser(unittest.TestCase):
    def setUp(self):
        self.parser = JsonParser()
    
    def test_simple_array(self):
        tokens = ['[', '1', ',', '2', ',', '3', ']']
        result = self.parser.parse_array(tokens)
        self.assertEqual(result, [1, 2, 3])
    
    def test_nested_array(self):
        tokens = ['[', '[', '1', ']', ',', '[', '2', ']', ']']
        result = self.parser.parse_array(tokens)
        self.assertEqual(result, [[1], [2]])
```
</context>

<thought_process>
1. Core Issue:
   - Bug: ValueError raised for empty arrays
   - Expected: Return empty list for '[]'
   - Examples: '[]' and '[[], []]'
   - Current code explicitly rejects empty arrays

2. Context Analysis:
   - Bug location identified in parse_array method
   - Existing test structure uses unittest
   - Other array cases already tested
   - Empty array check is explicit blocker

3. Test Strategy:
   - Test both empty and nested empty arrays
   - Follow existing test class structure
   - Assert expected list structures
   - Use examples from bug report

4. Implementation Check:
   - Matches existing test style
   - Uses unittest assertions
   - Clear test documentation
   - Tests both reported cases
</thought_process>

<test_code>
import unittest
from json.parser import JsonParser

class TestJsonParser(unittest.TestCase):
    def test_empty_array_parsing(self):
        """Test that empty arrays are parsed correctly.
        Bug: Currently raises ValueError instead of returning []
        """
        parser = JsonParser()
        result = parser.parse_array(['[', ']'])
        self.assertEqual(result, [])
        
    def test_nested_empty_arrays(self):
        """Test parsing nested empty arrays.
        Bug: Currently raises ValueError instead of returning [[], []]
        """
        parser = JsonParser()
        result = parser.parse_array(['[', '[', ']', ',', '[', ']', ']'])
        self.assertEqual(result, [[], []])

if __name__ == '__main__':
    unittest.main()
</test_code>
</example>

Study the bug report, context, and any previous test attempts. Then write a minimal, self-contained test case following the thought process above.
'''.replace("{", "{{").replace("}", "}}")

  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {title}
Description: {body}
Comments: {comments}

Bug context summary:
{bug_context}

Previous bug reproducing file
{previous_bug_reproducing_file}

Previous bug reproducing fail log
{previous_bug_reproducing_fail_log}
"""

  def __init__(self, model: BaseChatModel, kg: KnowledgeGraph):
    self.tools = self._init_tools(kg.get_local_path())
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.model_with_tools = model.bind_tools(self.tools)
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

    return tools

  def format_human_message(self, state: BugReproductionState):
    previous_bug_reproducing_file = ""
    if "reproduced_bug_file" in state and state["reproduced_bug_file"]:
      previous_bug_reproducing_file = state["reproduced_bug_file"]
    previous_bug_reproducing_fail_log = ""
    if "reproduced_bug_failure_log" in state and state["reproduced_bug_failure_log"]:
      previous_bug_reproducing_fail_log = state["reproduced_bug_failure_log"]
    return self.HUMAN_PROMPT.format(
      title=state["issue_title"],
      body=state["issue_body"],
      comments=state["issue_comments"],
      bug_context=state["bug_context"],
      previous_bug_reproducing_file=previous_bug_reproducing_file,
      previous_bug_reproducing_fail_log=previous_bug_reproducing_fail_log,
    )

  def __call__(self, state: BugReproductionState):
    human_message = HumanMessage(self.format_human_message(state))
    message_history = [self.system_prompt, human_message] + state["bug_reproducing_write_messages"]
    response = self.model_with_tools.invoke(message_history)

    self._logger.debug(f"BugReproducingWriteNode response:\n{response}")
    return {"bug_reproducing_write_messages": [response]}
