from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prometheus.app.api import issue
from prometheus.app.exception_handler import register_exception_handlers
from prometheus.lang_graph.graphs.issue_state import IssueType

app = FastAPI()
register_exception_handlers(app)
app.include_router(issue.router, prefix="/issue", tags=["issue"])
client = TestClient(app)


@pytest.fixture
def mock_service():
    service = mock.MagicMock()
    app.state.service = service
    yield service


def test_answer_issue(mock_service):
    mock_service["knowledge_graph_service"].exists_knowledge_graph.return_value = True
    mock_service["issue_service"].answer_issue.return_value = (
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
        "code": 200,
        "message": "success",
        "data": {
            "remote_branch_name": "feature/fix-42",
            "patch": "test patch",
            "passed_reproducing_test": True,
            "passed_build": True,
            "passed_existing_test": True,
            "issue_response": "Issue fixed",
        },
    }


def test_answer_issue_no_repository(mock_service):
    mock_service["knowledge_graph_service"].exists.return_value = False

    response = client.post(
        "/issue/answer",
        json={"issue_number": 42, "issue_title": "Test Issue", "issue_body": "Test description"},
    )

    assert response.status_code == 404


def test_answer_issue_invalid_container_config(mock_service):
    mock_service["knowledge_graph_service"].exists.return_value = True

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


def test_answer_issue_with_container(mock_service):
    mock_service["knowledge_graph_service"].exists.return_value = True
    mock_service["issue_service"].answer_issue.return_value = (
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
    mock_service["issue_service"].answer_issue.assert_called_once_with(
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
