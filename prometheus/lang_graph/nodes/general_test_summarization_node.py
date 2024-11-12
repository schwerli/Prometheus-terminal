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
3. fail_log: Comprehensive record of test failures including:
   - Complete test failure messages and stack traces
   - Test names and descriptions that failed
   - Error messages and assertions that failed
   - Relevant code snippets where failures occurred
   - Test execution summary (e.g., "2 failed, 3 passed")
   Return an empty string ONLY if all tests passed successfully

When analyzing commands:
- Focus on essential test execution steps
- Include test dependency installation commands
- List commands in execution order
- Note any required environment setup for testing
- Include commands for different test types (unit, integration, etc.)

When capturing fail logs:
- Include ALL test failure information from execution output
- Show complete stack traces and error messages
- Capture test summaries and statistics
- Include surrounding context of failures when available
- Format the output to maintain readability

The input will contain messages showing the agent's attempts and their results.

Example 1: Python Project with Test Failures
Input:
Assistant internal thought: Looking for Python test files
Assistant executed command: {"command": "find . -name 'test_*.py'"}
Command output: ./tests/test_authentication.py
./tests/integration/test_api.py
Assistant internal thought: Found test files. Checking configuration
Assistant executed command: {"command": "cat pytest.ini"}
Command output: [pytest]
addopts = -v --cov=auth
testpaths = tests
Assistant internal thought: Installing dependencies and running tests
Assistant executed command: {"command": "pip install -r requirements.txt"}
Command output: Successfully installed pytest-7.3.1 pytest-cov-4.0.0
Assistant executed command: {"command": "python -m pytest"}
Command output: ============================= test session starts ==============================
platform linux -- Python 3.8.10, pytest-7.3.1, pluggy-1.0.0
collected 5 tests

tests/test_authentication.py::test_valid_login PASSED                      [ 20%]
tests/test_authentication.py::test_invalid_password PASSED                 [ 40%]
tests/test_authentication.py::test_account_lockout FAILED                 [ 60%]
tests/integration/test_api.py::test_api_auth FAILED                       [ 80%]
tests/integration/test_api.py::test_api_timeout PASSED                    [100%]

FAILED tests/test_authentication.py::test_account_lockout
    def test_account_lockout():
        for _ in range(5):
            auth.attempt_login("user", "wrong")
>       assert auth.is_account_locked("user") == True
E       AssertionError: assert False == True
E       +  where False = <function is_account_locked at 0x7f9b8c2e4f70>('user')

tests/test_authentication.py:45: AssertionError

FAILED tests/integration/test_api.py::test_api_auth
    def test_api_auth():
>       response = client.post("/api/login", json={"username": "test", "password": "test123"})
E       requests.exceptions.ConnectionError: HTTPConnectionError(Connection refused)
E       During handling of the above exception, another exception occurred:
E       ConnectionRefusedError: [Errno 111] Connection refused

tests/integration/test_api.py:23: ConnectionRefusedError
======================= 2 failed, 3 passed in 1.25s ===========================

Output:
{
    "exist_test": true,
    "command_summary": "Project uses pytest framework with coverage reporting. Required steps:\n1. Install dependencies: pip install -r requirements.txt\n2. Run tests: python -m pytest\nTests include unit tests in ./tests and integration tests in ./tests/integration, configured via pytest.ini for coverage reporting.",
    "fail_log": "============================= test session starts ==============================\nplatform linux -- Python 3.8.10, pytest-7.3.1, pluggy-1.0.0\ncollected 5 tests\n\nFAILED tests/test_authentication.py::test_account_lockout\n    def test_account_lockout():\n        for _ in range(5):\n            auth.attempt_login(\"user\", \"wrong\")\n>       assert auth.is_account_locked(\"user\") == True\nE       AssertionError: assert False == True\nE       +  where False = <function is_account_locked at 0x7f9b8c2e4f70>('user')\n\ntests/test_authentication.py:45: AssertionError\n\nFAILED tests/integration/test_api.py::test_api_auth\n    def test_api_auth():\n>       response = client.post(\"/api/login\", json={\"username\": \"test\", \"password\": \"test123\"})\nE       requests.exceptions.ConnectionError: HTTPConnectionError(Connection refused)\nE       During handling of the above exception, another exception occurred:\nE       ConnectionRefusedError: [Errno 111] Connection refused\n\ntests/integration/test_api.py:23: ConnectionRefusedError\n======================= 2 failed, 3 passed in 1.25s ==========================="
}

Example 2: JavaScript Project with Passing Tests
Input:
Assistant internal thought: Checking package.json for test configuration
Assistant executed command: {"command": "cat package.json"}
Command output: {
  "name": "payment-processor",
  "scripts": {
    "test": "jest --coverage",
    "test:watch": "jest --watch"
  },
  "devDependencies": {
    "jest": "^27.0.0",
    "@testing-library/react": "^12.0.0"
  }
}
Assistant internal thought: Found Jest configuration. Looking for test files
Assistant executed command: {"command": "find . -name '*.test.js' -o -name '*.spec.js'"}
Command output: ./src/utils/currency.test.js
./src/components/PaymentForm.test.js
Assistant internal thought: Installing dependencies and running tests
Assistant executed command: {"command": "npm install"}
Command output: added 342 packages in 15s
Assistant executed command: {"command": "npm test"}
Command output: PASS src/utils/currency.test.js
  Currency Formatter
    ✓ formats USD correctly (2ms)
    ✓ formats EUR with proper symbol (1ms)
    ✓ handles decimal places correctly (1ms)
PASS src/components/PaymentForm.test.js
  Payment Form Component
    ✓ renders all form fields (23ms)
    ✓ validates card number format (15ms)
    ✓ shows error for invalid expiry date (12ms)
    ✓ submits form with valid data (45ms)

Test Suites: 2 passed, 2 total
Tests:       7 passed, 7 total
Snapshots:   0 total
Time:        2.145s
Coverage: 94.32%

Output:
{
    "exist_test": true,
    "command_summary": "Project uses Jest framework with React Testing Library. Required steps:\n1. Install dependencies: npm install\n2. Run tests with coverage: npm test\nTests cover both utility functions and React components with a coverage threshold of 94.32%.",
    "fail_log": ""
}

Example 3: Project without Tests
Input:
Assistant internal thought: Checking for common test files and configurations
Assistant executed command: {"command": "find . -type f -name '*test*' -o -name '*spec*'"}
Command output: 
Assistant internal thought: No test files found. Checking build configurations
Assistant executed command: {"command": "cat package.json"}
Command output: {
  "name": "static-website",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.17.1",
    "lodash": "^4.17.21"
  }
}
Assistant executed command: {"command": "find . -name 'pytest.ini' -o -name 'jest.config.js'"}
Command output: 

Output:
{
    "exist_test": false,
    "command_summary": "No test framework detected. Project is a basic Node.js application with Express and utility dependencies, but lacks any test files, test frameworks, or test-related configuration.",
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
