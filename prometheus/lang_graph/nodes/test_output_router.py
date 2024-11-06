from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class TestClassification(BaseModel):
  fixed_issue: bool = Field(description="Whether the issue is fixed or not")


class TestOutputRouter:
  SYS_PROMPT = """\
You are a QA analysis agent tasked with determining if a specific bug has been successfully fixed by comparing test outputs from before and after a patch was applied.

CORE RESPONSIBILITIES:
1. Issue Understanding
   - Carefully analyze the GitHub issue description
   - Identify the specific symptoms and behaviors that need to be fixed
   - Note any specific test cases or conditions mentioned in the issue

2. Test Output Analysis
   - Compare pre-patch and post-patch test outputs
   - Look specifically for changes related to the described issue
   - Identify if the problematic behavior has been resolved
   - Check for any new issues or regressions introduced

3. Verification Criteria
   - The issue is considered fixed ONLY if:
     a) The specific problem described in the GitHub issue is no longer present
     b) The test outputs show clear evidence of the fix
     c) No new critical issues are introduced
   - The issue is NOT considered fixed if:
     a) The original problem persists in any form
     b) The test output is inconclusive
     c) The fix appears to address a different issue
     d) New critical issues are introduced

4. Decision Making
   - Make a boolean decision based strictly on whether the SPECIFIC issue is fixed
   - Provide clear reasoning for the decision
   - Do not consider partial fixes as successful
   - Do not be misled by unrelated test improvements

Remember: Focus solely on whether the specific issue described in the GitHub issue is fixed, not on general improvements or other changes in the test output.
"""

  HUMAN_PROMPT = """\
Pre-patch test output:
{before_test_output}

Post-patch test output:
{after_test_output}

GitHub issue body:
{issue_body}
"""

  def __init__(self, model: BaseChatModel):
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.structured_llm = model.with_structured_output(TestClassification)

  def format_human_message(self, state: IssueAnswerAndFixState) -> HumanMessage:
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(
        before_test_output=state["before_test_output"],
        after_test_output=state["after_test_output"],
        issue_body=state["issue_body"],
      )
    )
    return human_message

  def __call__(self, state: IssueAnswerAndFixState):
    message_history = [self.system_prompt, self.format_human_message(state)]
    response = self.structured_llm.invoke(message_history)
    return response.fixed_issue
