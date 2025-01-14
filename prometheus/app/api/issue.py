from typing import Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Request
from litellm import Field
from pydantic import BaseModel

from prometheus.lang_graph.graphs.issue_state import IssueType

router = APIRouter()


class IssueRequest(BaseModel):
  issue_number: int = Field(description="The number of the issue", examples=[42])
  issue_title: str = Field(
    description="The title of the issue", examples=["There is a memory leak"]
  )
  issue_body: str = Field(
    description="The description of the issue", examples=["foo/bar.c is causing a memory leak"]
  )
  issue_comments: Optional[Sequence[Mapping[str, str]]] = Field(
    default=None,
    description="Comments on the issue",
    examples=[
      [
        {"username": "user1", "comment": "I've experienced this issue as well."},
        {"username": "user2", "comment": "A potential fix is to adjust the memory settings."},
      ]
    ],
  )
  issue_type: IssueType = Field(
    default=IssueType.AUTO,
    description="The type of the issue, set to auto if you do not know",
    examples=[IssueType.AUTO],
  )
  run_build: Optional[bool] = Field(
    default=False,
    description="When editing the code, whenver we should run the build to verify the fix",
    examples=[False],
  )
  run_existing_test: Optional[bool] = Field(
    default=False,
    description="When editing the code, whenver we should run the existing test to verify the fix",
    examples=[False],
  )
  number_of_candidate_patch: Optional[int] = Field(
    default=4,
    description="When the patch is not verfied (through build or test), number of candidate patches we generate to select the best one",
    examples=[4],
  )
  dockerfile_content: Optional[str] = Field(
    default=None,
    description="Specify the containerized enviroment with dockerfile content",
    examples=["FROM python:3.11\nWORKDIR /app\nCOPY . /app"],
  )
  image_name: Optional[str] = Field(
    default=None,
    description="Specify the containerized enviroment with image name that should be pulled from dockerhub",
    examples=["python:3.11-slim"],
  )
  workdir: Optional[str] = Field(
    default=None,
    description="If you specified the container environment, you must also specify the workdir",
    examples=["/app"],
  )
  build_commands: Optional[Sequence[str]] = Field(
    default=None,
    description="If you specified dockerfile_content and run_build is True, you must also specify the build commands.",
    examples=[["pip install -r requirements.txt", "python -m build"]],
  )
  test_commands: Optional[Sequence[str]] = Field(
    default=None,
    description="If you specified dockerfile_content and run_test is True, you must also specify the test commands.",
    examples=[["pytest ."]],
  )
  push_to_remote: Optional[bool] = Field(
    default=False,
    description="When editing the code, whenver we should push the changes to a remote branch",
    examples=[True],
  )


@router.post(
  "/answer/",
  summary="Process and generate a response for an issue",
  description="Analyzes an issue, generates patches if needed, runs optional builds and tests, and can push changes to a remote branch.",
  response_description="Returns the patch, test results, and issue response",
)
def answer_issue(issue: IssueRequest, request: Request):
  if not request.app.state.service_coordinator.exists_knowledge_graph():
    raise HTTPException(
      status_code=404,
      detail="A repository is not uploaded, use /repository/ endpoint to upload one",
    )

  if issue.dockerfile_content or issue.image_name:
    if issue.workdir is None:
      raise HTTPException(
        status_code=400,
        detail="workdir must be provided for user defined environment",
      )

  (
    remote_branch_name,
    patch,
    passed_reproducing_test,
    passed_build,
    passed_existing_test,
    issue_response,
  ) = request.app.state.service_coordinator.answer_issue(
    issue_number=issue.issue_number,
    issue_title=issue.issue_title,
    issue_body=issue.issue_body,
    issue_comments=issue.issue_comments if issue.issue_comments else [],
    issue_type=issue.issue_type,
    run_build=issue.run_build,
    run_existing_test=issue.run_existing_test,
    number_of_candidate_patch=issue.number_of_candidate_patch,
    dockerfile_content=issue.dockerfile_content,
    image_name=issue.image_name,
    workdir=issue.workdir,
    build_commands=issue.build_commands,
    test_commands=issue.test_commands,
    push_to_remote=issue.push_to_remote,
  )
  return {
    "patch": patch,
    "passed_reproducing_test": passed_reproducing_test,
    "passed_build": passed_build,
    "passed_existing_test": passed_existing_test,
    "issue_response": issue_response,
    "remote_branch_name": remote_branch_name,
  }
