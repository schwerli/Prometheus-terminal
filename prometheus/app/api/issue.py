from typing import Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class Issue(BaseModel):
  title: str
  body: str
  comments: Optional[Sequence[Mapping[str, str]]] = None


@router.post("/answer/")
def answer_issue(issue: Issue, request: Request):
  if not request.app.state.service_coordinator.exists_knowledge_graph():
    raise HTTPException(
      status_code=404,
      detail="A repository is not uploaded, use /repository/ endpoint to upload one",
    )

  response = request.app.state.service_coordinator.answer_issue(
    issue.title,
    issue.body,
    issue.comments if issue.comments else [],
  )
  return response
