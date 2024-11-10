"""Test execution analysis and summarization for software projects.

This module analyzes test execution attempts and provides structured summaries of
test frameworks, required commands, and failure information. It processes test
execution histories to determine test framework presence, extract test steps, and
identify any failures.
"""

import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class TestClassification(BaseModel):
  """Structured output model for test analysis results.

  Attributes:
    exist_test: Boolean indicating presence of a test framework.
    command_summary: Detailed description of test framework and required commands.
    fail_log: Detailed test failure logs and messages, empty string if all tests passed.
  """

  exist_test: bool = Field(
    description="Indicates if there is any test framework present in the project"
  )
  command_summary: str = Field(
    description="Summary of the test framework and list of commands required to run tests"
  )
  fail_log: str = Field(
    description="Contains the test failure logs if any tests failed, empty string if all passed"
  )


class GeneralTestSummarizationNode:
  """Analyzes and summarizes test execution attempts for software projects.

  This class processes test execution histories to provide structured analysis
  of test frameworks, required test steps, and any failures encountered. It can
  identify and analyze various testing frameworks including pytest, Jest, and others.
  """

  SYS_PROMPT = """\
You are a testing expert analyzing test execution history for software projects. You'll review
a history of commands executed by an agent that attempted to run the tests. Examine this test history to:

1. Determine if a test framework exists (looking for test files, pytest.ini, jest.config.js, etc.)
2. Analyze the test execution process and required commands
3. Identify any test failures and their causes

Provide three outputs:
1. exist_test: Boolean indicating if a test framework is present
2. command_summary: Concise description of the test setup and chronological list of commands needed for testing, including:
   - Type of test framework detected
   - Required test dependencies or setup steps
   - Sequence of test commands to execute
   - Test coverage and scope information
3. fail_log: If any tests failed, provide the relevant error logs and test failure details. Empty string if all tests passed

When analyzing commands:
- Focus on essential test execution steps
- Include test dependency installation commands
- List commands in execution order
- Note any required environment setup for testing
- Include commands for different test types (unit, integration, etc.)

When capturing fail logs:
- Include complete test failure messages
- Show which tests failed and why
- Include relevant stack traces
- Return empty string if all tests passed

The input will contain messages showing the agent's attempts and their results.

Example 1: Python Project with pytest
Input:
Assistant internal thought: Let me check if there are any Python test files
Assistant executed command: {"command": "find . -name 'test_*.py'"}
Command output: ./tests/test_calculator.py
./tests/integration/test_api.py
Assistant internal thought: Found test files. Let me check the test framework configuration
Assistant executed command: {"command": "cat pytest.ini"}
Command output: [pytest]
addopts = -v --cov=calculator
testpaths = tests
Assistant internal thought: This is a pytest project with coverage requirements. Installing dependencies and running tests
Assistant executed command: {"command": "pip install -r requirements.txt"}
Command output: Successfully installed pytest-7.3.1 pytest-cov-4.0.0
Assistant executed command: {"command": "python -m pytest"}
Command output: ============================= test session starts ==============================
platform linux -- Python 3.8.10, pytest-7.3.1, pluggy-1.0.0
collected 3 tests

tests/test_calculator.py::test_add PASSED                                   [ 33%]
tests/test_calculator.py::test_subtract PASSED                             [ 66%]
tests/test_calculator.py::test_multiply FAILED                             [100%]

FAILED tests/test_calculator.py::test_multiply
    def test_multiply():
>       assert calculator.multiply(3, 4) == 13
E       assert 12 == 13

tests/test_calculator.py:15: AssertionError
======================= 1 failed, 2 passed in 0.12s ===========================

Output:
{
    "exist_test": true,
    "command_summary": "Project uses pytest framework with coverage reporting. Required steps:\n1. Install dependencies: pip install -r requirements.txt\n2. Run tests: python -m pytest\nTests include unit tests in ./tests and integration tests in ./tests/integration with pytest.ini configuration for coverage reporting.",
    "fail_log": "FAILED tests/test_calculator.py::test_multiply\n    def test_multiply():\n>       assert calculator.multiply(3, 4) == 13\nE       assert 12 == 13\n\ntests/test_calculator.py:15: AssertionError"
}

Example 2: Project without Tests
Input:
Assistant internal thought: Checking for common test files and configurations
Assistant executed command: {"command": "find . -type f -name '*test*' -o -name '*spec*'"}
Command output: 
Assistant internal thought: No test files found. Let me check build configuration files
Assistant executed command: {"command": "cat package.json"}
Command output: {
  "name": "my-project",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.17.1"
  }
}

Output:
{
    "exist_test": false,
    "command_summary": "No test framework detected. Project contains basic Node.js configuration without any test files or test-related dependencies.",
    "fail_log": ""
}

Example 3: JavaScript Project with Jest
Input:
Assistant internal thought: Checking package.json for test configuration
Assistant executed command: {"command": "cat package.json"}
Command output: {
  "name": "utils-lib",
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch"
  },
  "devDependencies": {
    "jest": "^27.0.0"
  }
}
Assistant internal thought: Found Jest configuration. Looking for test files
Assistant executed command: {"command": "git ls-files '*.test.js'"}
Command output: src/utils.test.js
src/components/__tests__/button.test.js
Assistant internal thought: Installing dependencies and running tests
Assistant executed command: {"command": "npm install"}
Command output: added 234 packages in 12s
Assistant executed command: {"command": "npm test"}
Command output: PASS src/utils.test.js
  ✓ formats date correctly (3ms)
  ✓ validates email format (1ms)
PASS src/components/__tests__/button.test.js
  ✓ renders button correctly (5ms)
  ✓ handles click events (2ms)

Test Suites: 2 passed, 2 total
Tests:       4 passed, 4 total
Snapshots:   0 total
Time:        1.234s

Output:
{
    "exist_test": true,
    "command_summary": "Project uses Jest testing framework. Required steps:\n1. Install dependencies: npm install\n2. Run tests: npm test\nTests are organized in src/**/*.test.js pattern and component-specific tests in __tests__ directories.",
    "fail_log": ""
}
""".replace("{", "{{").replace("}", "}}")

  def __init__(self, model: BaseChatModel):
    """Initializes the GeneralTestSummarizationNode with an analysis model.

    Sets up the test summarizer with a prompt template and structured output
    model for analyzing test execution histories.

    Args:
      model: Language model instance that will be used for analyzing test
        histories and generating structured summaries. Must be a
        BaseChatModel implementation.
    """
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{test_history}")]
    )
    structured_llm = model.with_structured_output(TestClassification)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.general_test_summarization_node")

  def format_test_history(self, test_messages: Sequence[BaseMessage]):
    """Formats test execution messages into a structured history.

    Processes various message types (AI, Tool) into a chronological sequence
    of test execution steps and their outputs.

    Args:
      test_messages: Sequence of messages from test execution attempts.
        Can include AIMessage and ToolMessage types.

    Returns:
      List of formatted strings representing the test execution history,
      including internal thoughts, executed commands, and their outputs.
    """
    formatted_messages = []
    for message in test_messages:
      if isinstance(message, AIMessage):
        if message.content:
          formatted_messages.append(f"Assistant internal thought: {message.content}")
        if message.additional_kwargs and message.additional_kwargs["tool_calls"]:
          for tool_call in message.additional_kwargs["tool_calls"]:
            formatted_messages.append(f"Assistant executed command: {tool_call.function.arguments}")
      elif isinstance(message, ToolMessage):
        formatted_messages.append(f"Command output: {message.content}")
    return formatted_messages

  def __call__(self, state: IssueAnswerAndFixState):
    """Processes test state to generate structured test analysis.

    Analyzes the test execution history to determine test framework presence,
    required commands, and any failures encountered.

    Args:
      state: Current state containing test execution messages and history.

    Returns:
      Dictionary that updates the state containing:
      - exist_test: Boolean indicating if a test framework exists
      - test_command_summary: String describing test framework and required commands
      - test_fail_log: String containing test failure details (empty if all passed)
    """
    test_history = "\n".join(self.format_test_history(state["test_messages"]))
    response = self.model.invoke({"test_history": test_history})
    self._logger.debug(f"GeneralTestSummarizationNode response:\n{response}")
    return {
      "exist_test": response.exist_test,
      "test_command_summary": response.command_summary,
      "test_fail_log": response.fail_log,
    }
