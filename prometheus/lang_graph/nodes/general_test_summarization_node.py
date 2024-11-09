import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class TestClassification(BaseModel):
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
"""

  def __init__(self, model: BaseChatModel):
    self.model_with_structured_output = model.with_structured_output(TestClassification)
    self.sys_prompt = SystemMessage(self.SYS_PROMPT)
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.general_test_summarization_node")

  def format_test_history(self, test_messages: Sequence[BaseMessage]):
    formatted_messages = []
    for message in test_messages:
      if isinstance(message, AIMessage):
        formatted_messages.append(f"Assistant message: {message.content}")
      elif isinstance(message, ToolMessage):
        formatted_messages.append(f"Tool message: {message.content}")
    return formatted_messages

  def __call__(self, state: IssueAnswerAndFixState):
    human_message = HumanMessage("\n".join(self.format_test_history(state["test_messages"])))
    message_history = [self.sys_prompt + human_message]
    self._logger.debug(f"GeneralTestSummarizationNode human message:\n{human_message.content}")
    response = self.model_with_structured_output.invoke(message_history)
    self._logger.debug(f"GeneralTestSummarizationNode response:\n{response}")
    return {
      "exist_test": response["exist_test"],
      "test_command_summary": response["command_summary"],
      "test_fail_log": response["fail_log"],
    }
