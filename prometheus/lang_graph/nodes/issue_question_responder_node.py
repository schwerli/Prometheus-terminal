import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.lang_graph.graphs.issue_state import IssueState
from prometheus.utils.issue_util import format_issue_comments


class IssueQuestionResponderNode:
  SYS_PROMPT = """\
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

  def format_human_message(self, state: IssueState) -> HumanMessage:
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(
        title=state["issue_title"],
        body=state["issue_body"],
        comments=format_issue_comments(state["issue_comments"]),
        question_context=state["question_context"],
      )
    )
    return human_message

  def __call__(self, state: IssueState):
    messages = [
      self.system_prompt,
      self.format_human_message(state),
    ]
    response = self.model.invoke(messages)
    self._logger.debug(f"IssueQuestionResponderNode reponse:\n{response}")
    return {"issue_response": response.content}
