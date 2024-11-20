from typing import Mapping, Sequence, TypedDict


class IssueClassificationState(TypedDict):
  # Attributes provided by the user
  issue_title: str
  issue_body: str
  issue_comments: Sequence[Mapping[str, str]]

  # Attributes generated and by the subgraph
  classification_context: str
  issue_type: str
