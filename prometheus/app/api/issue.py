from typing import Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Request
from litellm import Field
from pydantic import BaseModel

router = APIRouter()


class IssueAnswerAndFixRequest(BaseModel):
  number: int = Field(description="The number of the issue", examples=[42])
  title: str = Field(description="The title of the issue", examples=["There is a memory leak"])
  body: str = Field(
    description="The description of the issue", examples=["foo/bar.c is causing a memory leak"]
  )
  comments: Optional[Sequence[Mapping[str, str]]] = Field(
    default=None,
    description="Comments on the issue",
    examples=[
      [
        {"username": "user1", "comment": "I've experienced this issue as well."},
        {"username": "user2", "comment": "A potential fix is to adjust the memory settings."},
      ]
    ],
  )
  only_answer: Optional[bool] = Field(
    default=True, description="Only answer the issue without editing the code", examples=[True]
  )
  run_build: Optional[bool] = Field(
    default=True,
    description="When editing the code, whenver we should run the build to verify the fix",
    examples=[True],
  )
  run_test: Optional[bool] = Field(
    default=True,
    description="When editing the code, whenver we should run the test to verify the fix",
    examples=[True],
  )


@router.post(
  "/answer_and_fix/",
  summary="Answer and optionally also fix an issue",
  description="""
    Use Prometheus to answer and optionally also fix an issue.
    
    When only_answer is false, Promestheus will edit the code to fix the issue, push the change to a remote branch.
    run_build or run_test when set to True, will use build/test to verify the correctness of the genenerated changes.
    """,
  response_description="""
    A response containing the answer and the remote branch name.
    """,
)
def answer_and_fix_issue(issue: IssueAnswerAndFixRequest, request: Request):
  if not request.app.state.service_coordinator.exists_knowledge_graph():
    raise HTTPException(
      status_code=404,
      detail="A repository is not uploaded, use /repository/ endpoint to upload one",
    )

  issue_response, remote_branch_name = request.app.state.service_coordinator.answer_and_fix_issue(
    issue.number,
    issue.title,
    issue.body,
    issue.comments if issue.comments else [],
    issue.only_answer,
    issue.run_build,
    issue.run_test,
  )
  return {"issue_response": issue_response, "remote_branch_name": remote_branch_name}
