from operator import add
from typing import Annotated, Mapping, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from prometheus.models.context import Context


class IssueNotVerifiedBugState(TypedDict):
    issue_title: str
    issue_body: str
    issue_comments: Sequence[Mapping[str, str]]
    number_of_candidate_patch: int

    max_refined_query_loop: int

    bug_fix_query: str
    bug_fix_context: Sequence[Context]

    issue_bug_analyzer_messages: Annotated[Sequence[BaseMessage], add_messages]
    edit_messages: Annotated[Sequence[BaseMessage], add_messages]

    edit_patches: Annotated[Sequence[str], add]

    final_patch: str

    run_regression_test: bool
