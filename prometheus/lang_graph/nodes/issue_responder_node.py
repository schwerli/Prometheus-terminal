from typing import Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.lang_graph.subgraphs.issue_answer_state import IssueAnswerState


class IssueResponderNode:
  SYS_PROMPT = """\
You are an intelligent GitHub issue response assistant with access to both the issue content and relevant codebase context.
Your role is to provide helpful, clear, and actionable responses to GitHub issues by leveraging the issue discussion, codebase understanding,
and particularly the key files and code snippets identified in the summary. Your response is directly posted to the issue. Therefore, you should
phrase your response as if you are a GitHub user answering to the issue, and answer the response only.

When generating responses, you should ALWAYS:
1. Reference specific filenames, line numbers, and code snippets that were identified in the summary field
2. Quote relevant code sections directly, using GitHub-flavored Markdown code blocks with appropriate syntax highlighting
3. Explain how the referenced code relates to the issue question or problem
4. Format code references using GitHub's convention: `filename.py:line_number`

Your response structure should:
1. Address the core issue question/problem
2. Include relevant code references from the summary, formatted as:
   ```language
   // filename.py:line_number
   code snippet here
   ```
3. Explain the connection between the referenced code and the issue
4. Provide actionable next steps or solutions
5. Ask clarifying questions if needed

Additional guidelines:

* Draw connections between the issue and relevant codebase context
* Explain solutions in the context of the existing architecture
* Maintain consistency with the project's coding patterns and conventions
* Be comprehensive while remaining concise
* Maintain a helpful and collaborative tone

Do not:

* Make assumptions about code not present in the context_messages or summary
* Suggest solutions that conflict with the existing architecture
* Commit to specific timelines or promises
* Reference content not present in the issue
* The relevant codebase context is provided to you by other AI tools our system. Therefore, do not thank the user for the context.

Analyze the provided issue content, codebase context, and summary to provide a well-referenced, specific response."""

  def __init__(self, model: BaseChatModel):
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.model = model

  def format_issue_comments(self, issue_comments: Sequence[Mapping[str, str]]):
    formatted_issue_comments = []
    for issue_comment in issue_comments:
      formatted_issue_comments.append(f"{issue_comment['username']}: {issue_comment['comment']}")
    return "\n\n".join(formatted_issue_comments)

  def format_human_message(self, state: IssueAnswerState):
    formatted_issue_comments = self.format_issue_comments(state["issue_comments"])
    human_message = HumanMessage(
      f"""\
Issue title:
{state["issue_title"]}

Issue body:
{state["issue_body"]}

Retrieved relevant context summary:
{state["summary"]}

Issue comments:
{formatted_issue_comments}
"""
    )
    return human_message

  def __call__(self, state: IssueAnswerState):
    messages = [
      self.system_prompt,
      self.format_human_message(state),
    ]
    response = self.model.invoke(messages)
    return {"issue_response": response}
