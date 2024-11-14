import logging
from typing import Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


class RequireEditClassifierOutput(BaseModel):
  require_edit: bool = Field(
    description="If resolving the issue requires editing the files (code, documentation, etc.), "
    "or it just can answered with the information from the context retrieved from the codebase.",
  )


class RequireEditClassifierNode:
  SYS_PROMPT = """\
You are an agent that determines whether an issue can be resolved by providing information from the existing codebase context, or if it requires making changes to files (code, documentation, etc.).

Your task is to analyze the provided issue information (title, body, and comments) along with the relevant codebase context, and determine if:

1. The issue can be resolved by simply providing information or explanations using the available context (require_edit = false)
   Examples:
   - Questions about how certain features work
   - Requests for clarification about existing functionality
   - Help understanding error messages
   - Questions about configuration options
   - Inquiries about API usage

2. The issue requires changes to files in the codebase (require_edit = true)
   Examples:
   - Bug reports that need code fixes
   - Feature requests
   - Documentation updates or corrections
   - Performance improvements
   - Security vulnerabilities that need patching
   - Configuration file changes

You will receive:
- ISSUE INFORMATION: Contains the issue title, body, and any comments from users
- CODEBASE CONTEXT: Contains relevant information retrieved from the codebase that might help resolve the issue

Example 1 - Information Request:
ISSUE INFORMATION:
Title: How to configure logging level?
Body: I'm trying to understand how to change the logging level in production. Where is this configured?
Comments:
user1: Is this done through environment variables or a config file?

CODEBASE CONTEXT:
Found in config/logging.py:
Logging configuration can be set through LOGGING_LEVEL environment variable or in config.yaml under logging.level. Supported values are: DEBUG, INFO, WARNING, ERROR.

Expected Output:
{
  "require_edit": false
}
(Because the context contains the complete answer to the user's question)

Example 2 - Bug Report:
ISSUE INFORMATION:
Title: API returns 500 error on empty input
Body: When sending an empty request to /api/v1/process, the server crashes with 500 error instead of returning a proper validation error.
Comments:
user1: This happens consistently in production.
maintainer: Can you provide the error stack trace?
user1: Here's the trace: [stack trace showing null pointer exception]

CODEBASE CONTEXT:
Found in api/endpoints.py:
The /api/v1/process endpoint validates input but doesn't handle empty requests properly.

Expected Output:
{
  "require_edit": true
}
(Because code changes are needed to add proper input validation)

Example 3 - Feature Clarification:
ISSUE INFORMATION:
Title: Does the retry mechanism support exponential backoff?
Body: I'm implementing retries in my client code and wondering if the built-in retry mechanism supports exponential backoff.
Comments:
user1: Also interested in knowing the default retry settings.

CODEBASE CONTEXT:
Found in client/retry.py:
RetryStrategy class implements exponential backoff with default settings: max_retries=3, base_delay=1s, multiplier=2.
Usage example in comments shows how to configure custom retry parameters.

Expected Output:
{
  "require_edit": false
}
(Because the existing code already supports this feature and the context explains how to use it)

Example 4 - Documentation Update:
ISSUE INFORMATION:
Title: Authentication docs are outdated
Body: The authentication documentation still shows the old API key format. It should be updated to show the new Bearer token format.
Comments:
maintainer: You're right, this needs to be updated.

CODEBASE CONTEXT:
Found in docs/authentication.md:
Shows examples using the deprecated API key format.

Expected Output:
{
  "require_edit": true
}
(Because the documentation needs to be updated)

Respond only with a structured output containing the boolean field "require_edit".

Remember:
- Focus solely on determining if file changes are needed
- Don't attempt to solve the issue or suggest specific changes
- Consider only the information provided in the context
"""

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{context_info}")]
    )
    structured_llm = model.with_structured_output(RequireEditClassifierOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.require_edit_classifier_node")

  def format_issue_comments(self, issue_comments: Sequence[Mapping[str, str]]):
    """Formats a sequence of issue comments into a readable string.

    Combines multiple issue comments with their associated usernames into a
    formatted string suitable for inclusion in the response context.

    Args:
      issue_comments: Sequence of mappings containing 'username' and 'comment'
        keys for each issue comment.

    Returns:
      Formatted string containing all comments with usernames, separated by newlines.
    """
    formatted_issue_comments = []
    for issue_comment in issue_comments:
      formatted_issue_comments.append(f"{issue_comment['username']}: {issue_comment['comment']}")
    return "\n\n".join(formatted_issue_comments)

  def format_context_info(self, state: IssueAnswerAndFixState) -> str:
    context_info = f"""\
      ISSUE INFORMATION:
      Title: {state['issue_title']}
      Body: {state['issue_body']}
      Comments:
      {self.format_issue_comments(state['issue_comments'])}

      CODEBASE CONTEXT:
      {state['summary']}
    """
    return context_info

  def __call__(self, state: IssueAnswerAndFixState):
    context_info = self.format_context_info(state)
    response = self.model.invoke({"context_info": context_info})
    self._logger.debug(f"RequireEditClassifierNode response:\n{response}")
    return {"require_edit": response.require_edit}
