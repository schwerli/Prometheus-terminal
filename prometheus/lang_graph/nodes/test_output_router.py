from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class TestClassification(BaseModel):
  fixed_issue: bool = Field(description="Whether the issue is fixed or not")


class TestOutputRouter:
  SYS_PROMPT = """\
You are a QA analysis agent tasked with determining if a bug has been successfully fixed by comparing test outputs from before and after a patch was applied.

You will receive:
1. Pre-patch test output
2. Post-patch test output
3. Description of GitHub issue describing the bug

You task is to determine if the bug has been successfully fixed by outputting a single boolean value indicating if the bug has been fixed or not.
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
