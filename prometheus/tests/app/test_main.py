import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from prometheus.app.main import app


@pytest.fixture
def mock_service_coordinator():
    coordinator = MagicMock()
    coordinator.clear = MagicMock()
    coordinator.close = MagicMock()
    return coordinator


@pytest.fixture
def mock_initialize_services(mock_service_coordinator):
    with patch("prometheus.app.dependencies.initialize_services") as mock_init:
        mock_init.return_value = mock_service_coordinator
        yield mock_init


@pytest.fixture
def test_client(mock_initialize_services, mock_service_coordinator):
    with TestClient(app) as client:
        yield client
        # Verify cleanup was called
        mock_service_coordinator.clear.assert_called_once()
        mock_service_coordinator.close.assert_called_once()


def test_app_routers():
    """Test that the application has the expected routers configured"""
    routes = [route for route in app.routes if hasattr(route, "tags")]
    route_tags = {tag for route in routes for tag in route.tags}
    assert "repository" in route_tags
    assert "issue" in route_tags


def test_app_startup_shutdown(test_client, mock_initialize_services):
    """Test application startup initializes services correctly"""
    # The test_client fixture handles startup/shutdown
    mock_initialize_services.assert_called_once()