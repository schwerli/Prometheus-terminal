from fastapi import APIRouter, Request, HTTPException
from neo4j import GraphDatabase
from pydantic import BaseModel

from prometheus.agents import chat_history, context_provider_agent, message_types
from prometheus.configuration import config

from langchain_anthropic import ChatAnthropic

router = APIRouter()

class Query(BaseModel):
  query: str


@router.post("/send/")
def answer_issue(
  query: Query,
  request: Request
):
  if not hasattr(request.app.state, 'kg'):
    raise HTTPException(
      status_code=404,
      detail="A repository is not uploaded, use /repository/ endpoint to upload one",
    )

  if not hasattr(request.app.state, 'neo4j_driver'):
    request.app.state.neo4j_driver = GraphDatabase.driver(
      config.config["neo4j"]["uri"],
      auth=(config.config["neo4j"]["username"], config.config["neo4j"]["password"]),
    )

  if not hasattr(request.app.state, 'cp_agent'):
    llm = ChatAnthropic(
      model=config.config["anthropic"]["model"],
      temperature=config.config["anthropic"]["temperature"],
      max_tokens=config.config["anthropic"]["max_tokens"],
      api_key=config.config["anthropic"]["api_key"],
    )
    request.app.state.cp_agent = context_provider_agent.ContextProviderAgent(
      llm,
      request.app.state.kg,
      request.app.state.neo4j_driver
    )

  messages = chat_history.ChatHistory(10)
  messages.add_message(message_types.Message(role=message_types.Role.user, message=query.query))
  return request.app.state.cp_agent.get_response(messages)
