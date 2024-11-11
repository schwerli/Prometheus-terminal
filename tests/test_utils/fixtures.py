import shutil
import uuid

import pytest
from git import Repo
from testcontainers.neo4j import Neo4jContainer

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.neo4j.knowledge_graph_handler import KnowledgeGraphHandler
from tests.test_utils import test_project_paths

NEO4J_IMAGE = "neo4j:5.20.0"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"


@pytest.fixture(scope="session")
def neo4j_container_with_kg_fixture():
  kg = KnowledgeGraph(1000)
  kg.build_graph(test_project_paths.TEST_PROJECT_PATH)
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4J_PLUGINS", '["apoc"]').with_name(f"neo4j_container_with_kg_{uuid.uuid4().hex[:12]}")
  with container as neo4j_container:
    driver = neo4j_container.get_driver()
    handler = KnowledgeGraphHandler(driver, 100)
    handler.write_knowledge_graph(kg)
    yield neo4j_container, kg


@pytest.fixture(scope="session")
def empty_neo4j_container_fixture():
  container = Neo4jContainer(
    image=NEO4J_IMAGE, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
  ).with_env("NEO4J_PLUGINS", '["apoc"]').with_name(f"empty_neo4j_container_{uuid.uuid4().hex[:12]}")
  with container as neo4j_container:
    yield neo4j_container


@pytest.fixture(scope="function")
def git_repo_fixture():
  git_backup_dir = test_project_paths.TEST_PROJECT_PATH / ".git_backup"
  if git_backup_dir.exists():
    shutil.rmtree(git_backup_dir)
  shutil.copytree(test_project_paths.GIT_DIR, git_backup_dir)
  test_project_paths.GIT_DIR.rename(test_project_paths.TEST_PROJECT_PATH / ".git")

  repo = Repo(test_project_paths.TEST_PROJECT_PATH)
  yield repo
  repo.git.checkout("master")

  shutil.rmtree(test_project_paths.TEST_PROJECT_PATH / ".git")
  git_backup_dir.rename(test_project_paths.GIT_DIR)
