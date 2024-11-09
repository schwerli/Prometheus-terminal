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
  response = client.post("/issue/answer_and_fix/", json={"number": 1, "title": "title", "body": "body"})
  assert response.status_code == 404


def test_send(mock_service_coordinator):
  mock_service_coordinator.exists_knowledge_graph.return_value = True
  fake_response = "fake response"
  fake_branch_name = "fake_remote_branch_name"
  mock_response = (fake_response, fake_branch_name)
  mock_service_coordinator.answer_and_fix_issue.return_value = mock_response
  response = client.post("/issue/answer_and_fix/", json={"number": 1, "title": "title", "body": "body"})

  assert response.status_code == 200
  assert response.json() == {"issue_response": fake_response, "remote_branch_name": fake_branch_name}
