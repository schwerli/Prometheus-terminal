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
def mock_shared_state():
  mock_state = mock.MagicMock()
  app.state.shared_state = mock_state
  yield mock_state


def test_upload_local_repository(mock_shared_state):
  mock_shared_state.shared_state.upload_repository.return_value = None
  response = client.post(
    "/repository/local",
    json={"path": test_project_paths.TEST_PROJECT_PATH.absolute().as_posix()},
  )
  assert response.status_code == 200


def test_upload_fake_local_repository():
  response = client.post(
    "/repository/local",
    json={"path": "/foo/bar/baz.java"},
  )
  assert response.status_code == 404


def test_delete(mock_shared_state):
  mock_shared_state.kg_handler.knowledge_graph_exists.return_value = True
  mock_shared_state.clear_knowledge_graph.return_value = None
  response = client.get("repository/delete")
  assert response.status_code == 200
