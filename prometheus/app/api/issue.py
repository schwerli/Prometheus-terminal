from typing import Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class IssueAnswerAndFixRequest(BaseModel):
  number: int
  title: str
  body: str
  comments: Optional[Sequence[Mapping[str, str]]] = None
  only_answer: Optional[bool] = True
  run_build: Optional[bool] = True
  run_tests: Optional[bool] = True


@router.post("/answer_and_fix/")
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
    issue.run_tests,
  )
  return {"issue_response": issue_response, "remote_branch_name": remote_branch_name}
