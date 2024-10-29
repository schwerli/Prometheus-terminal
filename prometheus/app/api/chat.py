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

  conversation_id, response = request.app.state.shared_state.chat_with_context_provider(
    send_message.text, send_message.conversation_id
  )
  return {"conversation_id": conversation_id, "response": response}


@router.get("/all_conversation_ids/", response_model=list[str])
def get_all_conversation_ids(request: Request):
  return request.app.state.shared_state.get_all_conversation_ids()


@router.get("/conversation_messages/{conversation_id}", response_model=list[dict[str, str]])
def get_conversation_messages(conversation_id: str, request: Request):
  return request.app.state.shared_state.get_all_conversation_messages(conversation_id)
