from typing import Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Request
from litellm import Field
from pydantic import BaseModel

from prometheus.lang_graph.graphs.issue_state import IssueType

router = APIRouter()

# {
#  "issue_number": 42,
#  "issue_title": "Wrong behavior of calculator",
#  "issue_body": "When I am using your calculate application, 6 divided with 2 is equal to 12, which is not right!",
#  "issue_type": "bug",
#  "run_build": false,
#  "run_test": true,
#  "dockerfile_content": "FROM python:3.11\nWORKDIR /app\nCOPY . /app\nRUN pip install .",
#  "test_commands": ["pytest ."],
#  "workdir": "/app"
# }


class IssueAnswerRequest(BaseModel):
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


@router.post("/answer/")
def answer_and_fix_issue(issue: IssueAnswerRequest, request: Request):
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

    # Validate build commands
    if issue.run_build and issue.build_commands is None:
      raise HTTPException(
        status_code=400,
        detail="build_commands must be provided for user defined environment when run_build is True",
      )

    # Validate test commands
    if issue.run_existing_test and issue.test_commands is None:
      raise HTTPException(
        status_code=400,
        detail="test_commands must be provided for user defined environment when run_test is True",
      )

  request.app.state.service_coordinator.answer_issue(
    issue.issue_title,
    issue.issue_body,
    issue.issue_comments if issue.issue_comments else [],
    issue.issue_type,
    issue.run_build,
    issue.run_existing_test,
    issue.dockerfile_content,
    issue.image_name,
    issue.workdir,
    issue.build_commands,
    issue.test_commands,
  )
