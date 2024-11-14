import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class EditReviewerStructuredOutput(BaseModel):
  """Structured output model for reviewer comments.

  Attributes:
    reviewer_approved: Boolean if the reviewer approved the patch.
    reviewer_comments: The reviewer comments.
  """

  reviewer_approved: bool = Field(
    description="Indicates if there is any build system present in the project"
  )
  reviewer_comments: str = Field(description="The reviewer comments")


class EditReviewerStructuredNode:
  SYS_PROMPT = """\
You are a review output parser that converts unstructured code review feedback into a structured format.
You will receive review comments that follow this format:

VERDICT: [APPROVE or REQUEST_CHANGES]
SUMMARY: [Overview of assessment]
ISSUES: (optional section)
[List of issues]

Your task is to analyze these comments and output:
1. reviewer_approved: A boolean indicating if the patch is approved
   - Set to true if VERDICT is "APPROVE"
   - Set to false if VERDICT is "REQUEST_CHANGES"

2. reviewer_comments: A cleaned and formatted version of the review comments that:
   - Maintains the essential feedback
   - Preserves the original structure (Verdict, Summary, Issues, Notes)
   - Removes any redundant formatting or unnecessary whitespace
   - Ensures consistent formatting

Rules for parsing:
1. VERDICT interpretation:
   - "APPROVE" (or variations like "APPROVED", "LGTM") → reviewer_approved = true
   - "REQUEST_CHANGES" (or variations like "CHANGES_REQUESTED", "NEEDS_WORK") → reviewer_approved = false
   - If no clear verdict is found, default to false

2. Comment formatting:
   - Keep section headers in ALL CAPS
   - Maintain issue severity indicators [Critical], [Important], etc.
   - Preserve line breaks between sections
   - Remove extra whitespace or redundant formatting

Example input:
VERDICT: REQUEST_CHANGES

SUMMARY:
The patch addresses the core functionality but has a critical issue with error handling that could cause crashes in production.

ISSUES:
1. [Critical] The new error handler on line 45 swallows exceptions silently
2. [Important] The database connection isn't properly closed in the error path

Example output:
{
    "reviewer_approved": false,
    "reviewer_comments": "VERDICT: REQUEST_CHANGES\n\nSUMMARY:\nThe patch addresses the core functionality but has a critical issue with error handling that could cause crashes in production.\n\nISSUES:\n1. [Critical] The new error handler on line 45 swallows exceptions silently\n2. [Important] The database connection isn't properly closed in the error path"
}

Always maintain the original meaning and severity of the review while providing a clean, consistent structure.
""".replace("{", "{{").replace("}", "}}")

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{reviewer_comments}")]
    )
    structured_llm = model.with_structured_output(EditReviewerStructuredOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.edit_reviewer_structured_node")

  def __call__(self, state: IssueAnswerAndFixState):
    reviewer_comments = state["edit_reviewer_messages"][-1].content
    response = self.model.invoke({"reviewer_comments": reviewer_comments})
    self._logger.debug(f"EditReviewerStructuredNode response:\n{response}")

    return {
      "reviewer_approved": response.reviewer_approved,
      "reviewer_comments": response.reviewer_comments,
    }
