from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prometheus.app.api import issue
from prometheus.lang_graph.graphs.issue_state import IssueType

app = FastAPI()
app.include_router(issue.router, prefix="/issue", tags=["issue"])
client = TestClient(app)


@pytest.fixture
def mock_service_coordinator():
  service_coordinator = mock.MagicMock()
  app.state.service_coordinator = service_coordinator
  yield service_coordinator


def test_answer_issue(mock_service_coordinator):
  mock_service_coordinator.exists_knowledge_graph.return_value = True
  mock_service_coordinator.answer_issue.return_value = (
    "feature/fix-42",  # remote_branch_name
    "test patch",  # patch
    True,  # passed_reproducing_test
    True,  # passed_build
    True,  # passed_existing_test
    "Issue fixed",  # issue_response
  )

  response = client.post(
    "/issue/answer",
    json={
      "issue_number": 42,
      "issue_title": "Test Issue",
      "issue_body": "Test description",
    },
  )

  assert response.status_code == 200
  assert response.json() == {
    "remote_branch_name": "feature/fix-42",
    "patch": "test patch",
    "passed_reproducing_test": True,
    "passed_build": True,
    "passed_existing_test": True,
    "issue_response": "Issue fixed",
  }


def test_answer_issue_no_repository(mock_service_coordinator):
  mock_service_coordinator.exists_knowledge_graph.return_value = False

  response = client.post(
    "/issue/answer",
    json={"issue_number": 42, "issue_title": "Test Issue", "issue_body": "Test description"},
  )

  assert response.status_code == 404


def test_answer_issue_invalid_container_config(mock_service_coordinator):
  mock_service_coordinator.exists_knowledge_graph.return_value = True

  response = client.post(
    "/issue/answer",
    json={
      "issue_number": 42,
      "issue_title": "Test Issue",
      "issue_body": "Test description",
      "dockerfile_content": "FROM python:3.11",
      "workdir": None,
    },
  )

  assert response.status_code == 400


def test_answer_issue_with_container(mock_service_coordinator):
  mock_service_coordinator.exists_knowledge_graph.return_value = True
  mock_service_coordinator.answer_issue.return_value = (
    "feature/fix-42",
    "test patch",
    True,
    True,
    True,
    "Issue fixed",
  )

  test_payload = {
    "issue_number": 42,
    "issue_title": "Test Issue",
    "issue_body": "Test description",
    "dockerfile_content": "FROM python:3.11",
    "workdir": "/app",
    "build_commands": ["pip install -r requirements.txt"],
    "test_commands": ["pytest ."],
  }

  response = client.post("/issue/answer", json=test_payload)

  assert response.status_code == 200
  mock_service_coordinator.answer_issue.assert_called_once_with(
    issue_number=42,
    issue_title="Test Issue",
    issue_body="Test description",
    issue_comments=[],
    issue_type=IssueType.AUTO,
    run_build=False,
    run_existing_test=False,
    number_of_candidate_patch=4,
    dockerfile_content="FROM python:3.11",
    image_name=None,
    workdir="/app",
    build_commands=["pip install -r requirements.txt"],
    test_commands=["pytest ."],
    push_to_remote=False,
  )
