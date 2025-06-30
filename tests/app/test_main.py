from unittest.mock import MagicMock, patch

import pytest
from dynaconf.utils import DynaconfDict
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_settings():
    """
    Create a mock settings object and patch it before any tests run.
    The autouse=True ensures this fixture runs for all tests in the module.
    """
    mock_settings = {
        "LOGGING_LEVEL": "INFO",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password",
        "NEO4J_BATCH_SIZE": 1000,
        "KNOWLEDGE_GRAPH_MAX_AST_DEPTH": 5,
        "KNOWLEDGE_GRAPH_CHUNK_SIZE": 10000,
        "KNOWLEDGE_GRAPH_CHUNK_OVERLAP": 1000,
        "MAX_TOKEN_PER_NEO4J_RESULT": 5000,
        "ADVANCED_MODEL": "claude-3-5-sonnet-latest",
        "BASE_MODEL": "claude-3-5-haiku-latest",
        "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
        "WORKING_DIRECTORY": "/tmp",
        "GITHUB_ACCESS_TOKEN": "GITHUB_ACCESS_TOKEN",
        "POSTGRES_URI": "postgresql://postgres:password@localhost:5432/postgres?sslmode=disable",
        "MAX_TOKENS": 55000,
        "TEMPERATURE": 0.3,
    }

    # Create a DynaconfDict that properly implements attribute access
    settings_obj = DynaconfDict(mock_settings)

    # Ensure all settings are accessible as attributes
    for key, value in mock_settings.items():
        setattr(settings_obj, key, value)

    # Patch the settings before importing the app
    with patch("prometheus.configuration.config.settings", settings_obj):
        yield settings_obj


@pytest.fixture
def mock_dependencies():
    """Mock the service coordinator dependencies"""
    mock_coordinator = MagicMock()
    with patch("prometheus.app.dependencies.initialize_services", return_value=mock_coordinator):
        yield mock_coordinator


@pytest.fixture
def test_client(mock_settings, mock_dependencies):
    """Create a TestClient instance with mocked settings and dependencies"""
    # Import app here to ensure settings are properly mocked
    from prometheus.app.main import app

    with TestClient(app) as client:
        yield client


def test_app_initialization(test_client, mock_dependencies):
    """Test that the app initializes correctly with mocked dependencies"""
    assert test_client.app.state.service_coordinator is not None
    assert test_client.app.state.service_coordinator == mock_dependencies
