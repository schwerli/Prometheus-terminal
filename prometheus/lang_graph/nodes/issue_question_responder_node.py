import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.lang_graph.subgraphs.issue_question_state import IssueQuestionState
from prometheus.utils.issue_util import format_issue_comments


class IssueQuestionResponderNode:
  SYS_PROMPT = """\
You are an AI assistant that answers questions about issues. You will be provided with
issue information (title, body, comments) and your own analysis of the question context.
Use both the issue information and your contextual analysis to provide accurate, relevant responses.

GUIDELINES:
1. Use information from:
   - Issue title, body, and comments (user-provided content)
   - Question context (your own analysis)

2. Response Requirements:
   - Answer directly and concisely
   - Support answers with relevant quotes when helpful
   - If critical information is missing, state what's needed
   - Stay within the scope of available information
   - Use professional, neutral tone

Treat the question context as your own analytical insights about the issue, not as external information.
"""

  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {title}
Body: {body}
Comments:
{comments}

Retrieved question context:
{question_context}
"""

  def __init__(self, model: BaseChatModel):
    self.system_prompt = SystemMessage(self.SYS_PROMPT)
    self.model = model

    self._logger = logging.getLogger("prometheus.lang_graph.nodes.issue_question_responder_node")

  def format_human_message(self, state: IssueQuestionState) -> HumanMessage:
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(
        title=state["issue_title"],
        body=state["issue_body"],
        comments=format_issue_comments(state["issue_comments"]),
        question_context=state["question_context"],
      )
    )
    return human_message

  def __call__(self, state: IssueQuestionState):
    messages = [
      self.system_prompt,
      self.format_human_message(state),
    ]
    response = self.model.invoke(messages)
    self._logger.debug(f"IssueQuestionResponderNode reponse:\n{response}")
    return {"issue_response": response.content}
