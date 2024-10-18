from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from prometheus.agents import chat_history, message_types

router = APIRouter()


class Query(BaseModel):
  query: str


@router.post("/send/")
def answer_query(query: Query, request: Request):
  if not request.app.state.shared_state.has_knowledge_graph():
    raise HTTPException(
      status_code=404,
      detail="A repository is not uploaded, use /repository/ endpoint to upload one",
    )

  messages = chat_history.ChatHistory(10)
  messages.add_message(message_types.Message(role=message_types.Role.user, message=query.query))
  return request.app.state.shared_state.cp_agent.get_response(messages)
