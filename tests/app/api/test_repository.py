from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prometheus.app.api import repository
from tests.test_utils import test_project_paths

app = FastAPI()
app.include_router(repository.router, prefix="/repository", tags=["repository"])
client = TestClient(app)


@pytest.fixture
def mock_service_coordinator():
    service_coordinator = mock.MagicMock()
    app.state.service_coordinator = service_coordinator
    yield service_coordinator


def test_upload_local_repository(mock_service_coordinator):
    mock_service_coordinator.upload_local_repository.return_value = None
    response = client.get(
        "/repository/local",
        params={"local_repository": test_project_paths.TEST_PROJECT_PATH.absolute().as_posix()},
    )
    assert response.status_code == 200


def test_upload_fake_local_repository():
    response = client.get(
        "/repository/local/",
        params={"local_repository": "/foo/bar/"},
    )
    assert response.status_code == 404


def test_delete(mock_service_coordinator):
    mock_service_coordinator.exists_knowledge_graph.return_value = True
    mock_service_coordinator.clear.return_value = None
    response = client.get("repository/delete")
    assert response.status_code == 200
