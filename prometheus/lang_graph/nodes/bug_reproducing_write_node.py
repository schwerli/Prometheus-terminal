import functools
import logging

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.tools import file_operation


class BugReproducingWriteNode:
  SYS_PROMPT = '''\
You are a QA automation expert who writes focused test cases to reproduce software bugs.
Given an issue description, create a minimal test that demonstrates the problem.

Requirements:
- Include all necessary imports
- Must use the example from the issue if provided
- Write minimal number of assertions that fail now but will pass when fixed
- Keep tests minimal and focused

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

<similar_test_cases>
### Existing Array Parser Tests
```python
def test_parse_single_element_array(self):
    """Test parsing array with single element.
    Validates basic array parsing functionality.
    """
    tokens = ['[', '42', ']']
    result = self.parser.parse_array(tokens)
    self.assertEqual(result, [42])

def test_parse_nested_arrays(self):
    """Test parsing nested array structures.
    Ensures proper handling of array nesting.
    """
    tokens = ['[', '[', '1', ']', ',', '[', '2', ',', '3', ']', ']']
    result = self.parser.parse_array(tokens)
    self.assertEqual(result, [[1], [2, 3]])
```
</similar_test_cases>

<thought_process>
1. Similar Test Analysis:
   - Tests follow AAA pattern (Arrange-Act-Assert)
   - Each test has descriptive docstring
   - Parser instance created per test
   - Uses assertEqual for validation
   - Tests both simple and complex cases

2. Core Issue:
   - Bug: ValueError raised for empty arrays
   - Expected: Return empty list for '[]'
   - Examples: '[]' and '[[], []]'
   - Current code explicitly rejects empty arrays

3. Test Strategy:
   - Follow existing test structure and style
   - Test both empty and nested empty arrays
   - Match docstring format
   - Use same assertion pattern

4. Implementation Check:
   - Maintains test suite consistency
   - Clear documentation
   - Proper setup and teardown
   - Tests both reported cases
</thought_process>

<test_code>
import unittest
from json.parser import JsonParser

class TestJsonParser(unittest.TestCase):
    def test_empty_array_parsing(self):
        """Test parsing of empty array.
        Validates that empty arrays are handled correctly without raising errors.
        """
        parser = JsonParser()
        result = parser.parse_array(['[', ']'])
        self.assertEqual(result, [])
        
    def test_nested_empty_arrays(self):
        """Test parsing nested empty arrays.
        Ensures proper handling of multiple nested empty arrays.
        """
        parser = JsonParser()
        result = parser.parse_array(['[', '[', ']', ',', '[', ']', ']'])
        self.assertEqual(result, [[], []])

if __name__ == '__main__':
    unittest.main()
</test_code>
</example>

Study the bug report, similar test cases, and any previous test attempts. Then write a minimal, self-contained test
case following the thought process above. Pay special attention to maintaining consistency with the testing patterns shown in the similar test cases.
'''

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

  def __call__(self, state: BugReproductionState):
    message_history = [self.system_prompt] + state["bug_reproducing_write_messages"]
    response = self.model_with_tools.invoke(message_history)

    self._logger.debug(f"BugReproducingWriteNode response:\n{response}")
    return {"bug_reproducing_write_messages": [response]}
