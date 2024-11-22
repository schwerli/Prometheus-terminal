import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.bug_fix_verification_state import BugFixVerficationState


class BugFixVerifyStructureOutput(BaseModel):
  fixed_bug: bool = Field(description="Whether the bug is fixed (test passed)")
  reproducing_test_fail_log: str = Field(
    description="If the test failed, contains the complete test failure log"
  )


class BugFixVerifyStructuredNode:
  SYS_PROMPT = """\
You are a test result parser. Your only task is to:
1. Check if the test output contains exactly "Bug resolved"
2. If it doesn't contain "Bug resolved", copy the complete test output as the failure log

Return:
- fixed_bug: true ONLY if the test output contains exactly "Bug resolved", false otherwise
- reproducing_test_fail_log: empty string if "Bug resolved" is found, complete test output otherwise

Example for passing test (contains "Bug resolved"):
{
    "fixed_bug": true,
    "reproducing_test_fail_log": ""
}

Example for failing test (doesn't contain "Bug resolved"):
{
    "fixed_bug": false,
    "reproducing_test_fail_log": "<complete test output>"
}

Important: The fixed_bug field should ONLY be true if the exact string "Bug resolved" is found in the test output.
Any other output, even if it suggests success, should be treated as a failure.
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
