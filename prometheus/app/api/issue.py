from typing import Mapping, Optional, Sequence

from fastapi import APIRouter
from litellm import Field
from pydantic import BaseModel

from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import ResponseModeEnum

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
  response_mode: Optional[ResponseModeEnum] = Field(
    default=ResponseModeEnum.AUTO,
    description="The mode of response: auto (automatically determine whether to fix), only_answer (provide answer without changes), or answer_and_fix (provide answer and fix code)",
    examples=[ResponseModeEnum.AUTO, ResponseModeEnum.ONLY_ANSWER, ResponseModeEnum.ANSWER_AND_FIX],
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
