from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class TestClassification(BaseModel):
  exist_test: bool = Field(description="Indicates if there is any test framework present in the project")
  summary: str = Field(description="Summary of the test execution commands")
  successful: bool = Field(description="Whether all tests passed successfully")


class GeneralTestSummarizationNode:
  SYS_PROMPT = """\
You are a testing expert analyzing test execution history for software projects. You'll review
a history of commands executed by an agent that attempted to run the tests. Examine this test history to:

1. Determine if a test framework exists (looking for test files, pytest.ini, jest.config.js, etc.)
2. Assess if all tests passed (dependencies installed, tests executed, no unexpected failures)
3. Extract the test execution commands and results

Provide three outputs:
1. exist_test: Boolean indicating if a test framework is present
2. successful: Boolean indicating if all tests passed
3. summary: Structured test summary containing:
   - Test Framework: [type or "None"]
   - Status: [Passed/Failed/N/A] with results
   - Dependencies: [required packages/tools]
   - Environment: [required setup]
   - Steps: [commands with purpose]
   - Results: [total/passed/failed/skipped]
   - Notes: [warnings or issues]

Focus on command accuracy, test coverage, and failure analysis. The input will contain messages showing the agent's attempts and their results.
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
    return {"exist_test": response["exist_test"], "test_summary": response["summary"], "test_success": response["sucussful"]}
