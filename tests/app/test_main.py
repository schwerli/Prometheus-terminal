from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_dependencies():
    """Mock the service coordinator dependencies"""
    mock_coordinator = MagicMock()
    with patch("prometheus.app.dependencies.initialize_services", return_value=mock_coordinator):
        yield mock_coordinator


@pytest.fixture
def test_client(mock_dependencies):
    """Create a TestClient instance with mocked settings and dependencies"""
    # Import app here to ensure settings are properly mocked
    from prometheus.app.main import app

    with TestClient(app) as client:
        yield client


def test_app_initialization(test_client, mock_dependencies):
    """Test that the app initializes correctly with mocked dependencies"""
    assert test_client.app.state.service_coordinator is not None
    assert test_client.app.state.service_coordinator == mock_dependencies
