from pathlib import Path
from unittest.mock import Mock, create_autospec

import pytest

from prometheus.app.services.issue_service import IssueService
from prometheus.app.services.knowledge_graph_service import KnowledgeGraphService
from prometheus.app.services.llm_service import LLMService
from prometheus.app.services.neo4j_service import Neo4jService
from prometheus.app.services.repository_service import RepositoryService
from prometheus.lang_graph.graphs.issue_state import IssueType


@pytest.fixture
def mock_kg_service():
    service = create_autospec(KnowledgeGraphService, instance=True)
    service.kg = Mock(name="mock_knowledge_graph")
    service.kg.get_local_path.return_value = Path("/mock/path")
    return service


@pytest.fixture
def mock_repository_service():
    service = create_autospec(RepositoryService, instance=True)
    service.git_repo = Mock(name="mock_git_repo")
    return service


@pytest.fixture
def mock_neo4j_service():
    service = create_autospec(Neo4jService, instance=True)
    service.neo4j_driver = Mock(name="mock_neo4j_driver")
    return service


@pytest.fixture
def mock_llm_service():
    service = create_autospec(LLMService, instance=True)
    service.advanced_model = "gpt-4"
    service.base_model = "gpt-3.5-turbo"
    return service


@pytest.fixture
def issue_service(mock_kg_service, mock_repository_service, mock_neo4j_service, mock_llm_service):
    return IssueService(
        mock_kg_service,
        mock_repository_service,
        mock_neo4j_service,
        mock_llm_service,
        max_token_per_neo4j_result=1000,
        working_directory="/tmp/working_dir/",
    )


def test_answer_issue_with_general_container(issue_service, monkeypatch):
    # Setup
    mock_issue_graph = Mock()
    mock_issue_graph_class = Mock(return_value=mock_issue_graph)
    monkeypatch.setattr("prometheus.app.services.issue_service.IssueGraph", mock_issue_graph_class)

    mock_container = Mock()
    mock_general_container_class = Mock(return_value=mock_container)
    monkeypatch.setattr(
        "prometheus.app.services.issue_service.GeneralContainer", mock_general_container_class
    )

    # Mock output state for bug type
    mock_output_state = {
        "issue_type": IssueType.BUG,
        "edit_patch": "test_patch",
        "passed_reproducing_test": True,
        "passed_build": True,
        "passed_existing_test": True,
        "issue_response": "test_response",
    }
    mock_issue_graph.invoke.return_value = mock_output_state

    # Exercise
    result = issue_service.answer_issue(
        issue_number=-1,
        issue_title="Test Issue",
        issue_body="Test Body",
        issue_comments=[],
        issue_type=IssueType.BUG,
        run_build=True,
        run_existing_test=True,
        number_of_candidate_patch=1,
        run_reproduce_test=True,
    )

    # Verify
    mock_general_container_class.assert_called_once_with(
        issue_service.kg_service.kg.get_local_path()
    )
    mock_issue_graph_class.assert_called_once_with(
        advanced_model=issue_service.llm_service.advanced_model,
        base_model=issue_service.llm_service.base_model,
        kg=issue_service.kg_service.kg,
        git_repo=issue_service.repository_service.git_repo,
        neo4j_driver=issue_service.neo4j_service.neo4j_driver,
        max_token_per_neo4j_result=issue_service.max_token_per_neo4j_result,
        container=mock_container,
        build_commands=None,
        test_commands=None,
    )
    assert result == (None, "test_patch", True, True, True, "test_response")


def test_answer_issue_with_user_defined_container(issue_service, monkeypatch):
    # Setup
    mock_issue_graph = Mock()
    mock_issue_graph_class = Mock(return_value=mock_issue_graph)
    monkeypatch.setattr("prometheus.app.services.issue_service.IssueGraph", mock_issue_graph_class)

    mock_container = Mock()
    mock_user_container_class = Mock(return_value=mock_container)
    monkeypatch.setattr(
        "prometheus.app.services.issue_service.UserDefinedContainer", mock_user_container_class
    )

    # Mock output state for question type
    mock_output_state = {"issue_type": IssueType.QUESTION, "issue_response": "test_response"}
    mock_issue_graph.invoke.return_value = mock_output_state

    # Exercise
    result = issue_service.answer_issue(
        issue_number=-1,
        issue_title="Test Issue",
        issue_body="Test Body",
        issue_comments=[],
        issue_type=IssueType.QUESTION,
        run_build=True,
        run_existing_test=True,
        number_of_candidate_patch=1,
        dockerfile_content="FROM python:3.8",
        image_name="test-image",
        workdir="/app",
        build_commands=["pip install -r requirements.txt"],
        test_commands=["pytest"],
        run_reproduce_test=True,
    )

    # Verify
    mock_user_container_class.assert_called_once_with(
        issue_service.kg_service.kg.get_local_path(),
        "/app",
        ["pip install -r requirements.txt"],
        ["pytest"],
        "FROM python:3.8",
        "test-image",
    )
    assert result == (None, None, False, False, False, "test_response")
