from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class TestClassification(BaseModel):
  summary: str = Field(description="Summary of the test execution commands")
  successful: bool = Field(description="Whether all tests passed successfully")


class GeneralTestSummarizationNode:
  SYS_PROMPT = """\
You are a testing expert responsible for analyzing and summarizing test execution history for a software project.
Your goal is to determine if all tests passed successfully and extract the commands needed to run the tests.

Your responsibilities:
1. Analyze the test execution history to determine test success and identify working test commands
2. Extract and organize key information including:
   - Test framework/runner identified (pytest, jest, gtest, etc.)
   - Required test dependencies that were installed
   - Test execution commands that were run
   - Test environment setup requirements
   - Test configuration and flags used

Success Criteria:
Tests are considered successful if:
- All test dependencies were successfully installed
- Test runner executed without fatal errors
- All test cases passed (or expected failures were properly marked)
- No unexpected failures or errors occurred
- Test coverage requirements were met (if specified)

Output format:
You must output two pieces of information:
1. successful: A boolean indicating if all tests passed
2. summary: A structured summary of the test execution as follows:

Test Summary Structure:
1. Test Framework: [identified test framework/runner]
2. Test Status: [Passed/Failed - with summary of results]
3. Test Dependencies: [list of packages/tools needed for testing]
4. Environment Setup: [any environment variables or configuration needed]
5. Test Execution Steps:
   - Step 1: [command with purpose]
   - Step 2: [command with purpose]
   ...
6. Test Results:
   - Total Tests: [number]
   - Passed: [number]
   - Failed: [number]
   - Skipped/Ignored: [number]
7. Additional Notes: [important observations, warnings, or specific failures]

Guidelines:
- Carefully examine test output to determine success
- Include exact command syntax for reproducibility
- Note any specific test configuration requirements
- Document both passing and failing tests
- Include test runner version numbers where specified
- Note any platform-specific test requirements
- Pay attention to test coverage information if available
- Note any skipped or disabled tests

Remember:
- Be thorough in analyzing test results
- Include relevant error messages for failed tests
- Maintain exact command syntax
- Note any test dependencies or prerequisites
- Identify flaky or intermittent test failures
- Consider both unit and integration test results
"""

  def __init__(self, model: BaseChatModel):
    self.model_with_structured_output = model.with_structured_output(TestClassification)
    self.sys_prompt = SystemMessage(self.SYS_PROMPT)

  def format_test_history(self, test_messages: Sequence[BaseMessage]):
    formatted_messages = []
    for message in test_messages:
      if isinstance(message, AIMessage):
        formatted_messages.append(f"Assistant message: {message.content}")
      elif isinstance(message, ToolMessage):
        formatted_messages.append(f"Tool message: {message.content}")
    return formatted_messages

  def __call__(self, state: IssueAnswerAndFixState):
    message_history = [self.system_prompt] + HumanMessage(
      self.format_build_history(state["test_messages"])
    )
    response = self.model_with_structured_output.invoke(message_history)
    self._logger.debug(f"GeneralTestSummarizationNode response:\n{response}")
    return {"test_summary": response["summary"], "test_success": response["sucussful"]}
