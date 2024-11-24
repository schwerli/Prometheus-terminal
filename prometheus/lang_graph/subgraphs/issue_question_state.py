from typing import Mapping, Sequence, TypedDict


class IssueQuestionState(TypedDict):
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]

  question_context: str
  issue_response: str
