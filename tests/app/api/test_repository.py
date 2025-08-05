from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prometheus.app.api import repository
from prometheus.app.exception_handler import register_exception_handlers

app = FastAPI()
register_exception_handlers(app)
app.include_router(repository.router, prefix="/repository", tags=["repository"])
client = TestClient(app)


@pytest.fixture
def mock_service():
    service = mock.MagicMock()
    app.state.service = service
    yield service


def test_upload_repository(mock_service):
    mock_service["repository_service"].clone_github_repo.return_value = "/mock/path"
    response = client.get(
        "/repository/github/",
        params={
            "github_token": "mock_token",
            "https_url": "https://github.com/Pantheon-temple/Prometheus",
        },
    )
    assert response.status_code == 200


def test_upload_repository_at_commit(mock_service):
    mock_service["repository_service"].clone_github_repo.return_value = "/mock/path"
    response = client.get(
        "/repository/github_commit/",
        params={
            "github_token": "mock_token",
            "https_url": "https://github.com/Pantheon-temple/Prometheus",
            "commit_id": "0c554293648a8705769fa53ec896ae24da75f4fc",
        },
    )
    assert response.status_code == 200


def test_delete(mock_service):
    mock_service["knowledge_graph_service"].exists.return_value = True
    mock_service["knowledge_graph_service"].clear.return_value = None
    response = client.get("repository/delete")
    assert response.status_code == 200
