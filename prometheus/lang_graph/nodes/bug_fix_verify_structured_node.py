import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.bug_fix_verification_state import BugFixVerficationState


class BugFixVerifyStructureOutput(BaseModel):
  fixed_bug: bool = Field(
      description="Whether the bug is fixed, based on more flexible matching of test results"
  )
  reproducing_test_fail_log: str = Field(
      description="If the bug wasn't reproduced (fixed_bug is False), explains what went wrong with the reproduction attempt"
  )


class BugFixVerifyStructuredNode:
  SYS_PROMPT = """\
You are a structured output parser for bug fix verification results. Your role is to analyze
test execution logs and determine whether they successfully reproduce a reported bug.

Your task is to:
1. Parse the test execution output
2. Determine if the bug behavior is reproduced by analyzing error patterns and messages
3. Return a structured response containing:
   - fixed_bug (boolean): True if the test fails in a way that matches the reported bug
   - test_fail_log (string): If the bug wasn't reproduced, explain what went wrong with the reproduction attempt

Guidelines for analysis:
- A test SUCCESSFULLY REPRODUCES the bug (fixed_bug = True) if:
  * The error message contains similar keywords or patterns to the reported issue
  * The fundamental behavior matches the bug report
  * The failure occurs in the same component or code path
  * The error indicates the same underlying problem, even if error types differ

- When the bug is NOT reproduced (fixed_bug = False), the reproducing_test_fail_log should explain:
  * How the observed behavior differed from the reported bug
  * What type of error was expected vs what occurred
  * Whether the test passed when it should have failed
  * Any other reasons why the reproduction attempt failed

Important:
- Focus on the core bug behavior rather than exact error message matches
- Consider semantic similarity of errors rather than requiring exact type matches
- Look for patterns that indicate the same underlying issue
- Allow for reasonable variation in how errors manifest

Example responses:

For a passing test:
{
    "fixed_bug": true,
    "test_fail_log": ""
}

For a failing test:
{
    "fixed_bug": false,
    "reproducing_test_fail_log": ""Test failed but with wrong error type: got RuntimeError instead of expected ValueError."
}
""".replace("{", "{{").replace("}", "}}")

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
