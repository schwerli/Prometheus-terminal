from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prometheus.app.api import chat

app = FastAPI()
app.include_router(chat.router, prefix="/chat", tags=["chat"])
client = TestClient(app)


@pytest.fixture
def mock_shared_state():
  mock_state = MagicMock()
  app.state.shared_state = mock_state
  yield mock_state


def test_send_without_kg(mock_shared_state):
  mock_shared_state.has_knowledge_graph.return_value = False
  response = client.post("/chat/send", json={"text": "hello"})
  assert response.status_code == 404


def test_send(mock_shared_state):
  mock_conversation_id = "id"
  mock_response = "mock response"

  mock_shared_state.has_knowledge_graph.return_value = True
  mock_shared_state.chat_with_context_provider_agent.return_value = (
    mock_conversation_id,
    mock_response,
  )
  response = client.post("/chat/send", json={"text": "hello"})

  assert response.status_code == 200
  assert response.json() == {"conversation_id": mock_conversation_id, "response": mock_response}
