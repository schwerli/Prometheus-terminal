from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prometheus.app.api import issue

app = FastAPI()
app.include_router(issue.router, prefix="/issue", tags=["issue"])
client = TestClient(app)


@pytest.fixture
def mock_service_coordinator():
  service_coordinator = MagicMock()
  app.state.service_coordinator = service_coordinator
  yield service_coordinator


def test_answer_and_fix_issue(mock_service_coordinator):
  mock_service_coordinator.exists_knowledge_graph.return_value = False
  response = client.post("/issue/answer_and_fix/", json={"title": "title", "body": "body"})
  assert response.status_code == 404


def test_send(mock_service_coordinator):
  mock_service_coordinator.exists_knowledge_graph.return_value = True
  mock_response = "mock response"
  mock_service_coordinator.answer_and_fix_issue.return_value = mock_response
  response = client.post("/issue/answer_and_fix/", json={"title": "title", "body": "body"})

  assert response.status_code == 200
  assert response.json() == mock_response
