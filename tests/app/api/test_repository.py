from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prometheus.app.api import repository

app = FastAPI()
app.include_router(repository.router, prefix="/repository", tags=["repository"])
client = TestClient(app)


@pytest.fixture
def mock_service_coordinator():
    service_coordinator = mock.MagicMock()
    app.state.service_coordinator = service_coordinator
    yield service_coordinator


def test_delete(mock_service_coordinator):
    mock_service_coordinator.exists_knowledge_graph.return_value = True
    mock_service_coordinator.clear.return_value = None
    response = client.get("repository/delete")
    assert response.status_code == 200
