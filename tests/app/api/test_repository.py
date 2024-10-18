from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from neo4j import GraphDatabase
from testcontainers.neo4j import Neo4jContainer

from prometheus.app.api import repository
from prometheus.neo4j import knowledge_graph_handler
from tests.test_utils import test_project_paths

app = FastAPI()
app.include_router(repository.router, prefix="/repository", tags=["repository"])
client = TestClient(app)

NEO4J_IMAGE = "neo4j:5.20.0"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"


@pytest.fixture(scope="function")
def setup_neo4j_container():
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4J_PLUGINS", '["apoc"]')
  with container as neo4j_container:
    yield neo4j_container


def test_upload_local_repository(setup_neo4j_container):
  neo4j_container = setup_neo4j_container

  mock_config = {
    "neo4j": {
      "uri": neo4j_container.get_connection_url(),
      "username": NEO4J_USERNAME,
      "password": NEO4J_PASSWORD,
      "database": "neo4j",
      "batch_size": 100,
    },
    "knowledge_graph": {"max_ast_depth": 1000},
  }

  with mock.patch("prometheus.configuration.config.config", mock_config):
    response = client.post(
      "/repository/local",
      json={"path": test_project_paths.TEST_PROJECT_PATH.absolute().as_posix()},
    )
    assert response.status_code == 200

    kg_handler = knowledge_graph_handler.KnowledgeGraphHandler(
      neo4j_container.get_connection_url(),
      NEO4J_USERNAME,
      NEO4J_PASSWORD,
      "neo4j",
      100,
    )
    assert kg_handler.knowledge_graph_exists()
    kg_handler.close()


def test_delete(setup_neo4j_container):
  neo4j_container = setup_neo4j_container

  mock_config = {
    "neo4j": {
      "uri": neo4j_container.get_connection_url(),
      "username": NEO4J_USERNAME,
      "password": NEO4J_PASSWORD,
      "database": "neo4j",
      "batch_size": 100,
    },
    "knowledge_graph": {"max_ast_depth": 1000},
  }

  with mock.patch("prometheus.configuration.config.config", mock_config):
    response = client.post(
      "/repository/local",
      json={"path": test_project_paths.TEST_PROJECT_PATH.absolute().as_posix()},
    )
    assert response.status_code == 200

    kg_handler = knowledge_graph_handler.KnowledgeGraphHandler(
      neo4j_container.get_connection_url(),
      NEO4J_USERNAME,
      NEO4J_PASSWORD,
      "neo4j",
      100,
    )
    assert kg_handler.knowledge_graph_exists()

    response = client.get("/repository/delete")
    assert response.status_code == 200

    assert not kg_handler.knowledge_graph_exists()

    kg_handler.close()
