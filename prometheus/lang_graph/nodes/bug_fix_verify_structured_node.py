import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.bug_fix_verification_state import BugFixVerficationState


class BugFixVerifyStructureOutput(BaseModel):
  fixed_bug: bool = Field(
    description="Whenver the bug is fixed, ie. the bug exposing test is passing"
  )
  reproducing_test_fail_log: str = Field(
    description="The log from the test failure, empty if fixed_bug is True"
  )


class BugFixVerifyStructuredNode:
  SYS_PROMPT = """\
You are a structured output parser for bug fix verification results. Your role is to analyze
test execution logs and determine whether a bug has been successfully fixed.

Your task is to:
1. Parse the test execution output
2. Determine if the bug is fixed by checking if the test passes
3. Return a structured response containing:
   - fixed_bug (boolean): True if the test passes, False if it fails
   - test_fail_log (string): The relevant error/failure message if the test fails, empty string if test passes

Guidelines for analysis:
- A test is considered PASSING if:
  * It completes execution without errors
  * No assertion failures are reported
  * No exceptions are thrown
  * Exit code is 0 (if visible)

- A test is considered FAILING if:
  * Assertion failures occur
  * Exceptions are thrown
  * Error messages are present
  * Exit code is non-zero (if visible)

- When extracting test_fail_log:
  * Include only the relevant error message and stack trace
  * Trim unnecessary environment setup or teardown logs
  * Include any assertion messages that explain why the test failed
  * If multiple errors occur, include the first failure point

Do not:
- Try to interpret or fix the bugs
- Make assumptions about test behavior not evident in the logs
- Include passing test output in the test_fail_log
- Include system or environment setup messages in the test_fail_log

Example responses:

For a passing test:
{
    "fixed_bug": true,
    "test_fail_log": ""
}

For a failing test:
{
    "fixed_bug": false,
    "reproducing_test_fail_log": "AssertionError: Expected output 'Hello World' but got 'Hello'"
}
"""

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{bug_reproducing_logs}")]
    )
    structured_llm = model.with_structured_output(BugFixVerifyStructureOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_fix_verify_structured_node")

  def __call__(self, state: BugFixVerficationState):
    bug_fix_verify_message = state["bug_fix_verify_messages"][-1].content
    response = self.model.invoke({"bug_reproducing_logs": bug_fix_verify_message})
    self._logger.debug(f"BugFixVerifyStructuredNode response:\n{response}")

    return {
      "fixed_bug": response.fixed_bug,
      "reproducing_test_fail_log": response.reproducing_test_fail_log,
    }
