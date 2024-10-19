from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class SendMessage(BaseModel):
  text: str
  conversation_id: Optional[str] = None


@router.post("/send/")
def answer_query(send_message: SendMessage, request: Request):
  if not request.app.state.shared_state.has_knowledge_graph():
    raise HTTPException(
      status_code=404,
      detail="A repository is not uploaded, use /repository/ endpoint to upload one",
    )


  return request.app.state.shared_state.cp_agent.get_response(send_message.text)
