import pytest

from prometheus.app.services.database_service import DatabaseService
from prometheus.app.services.user_service import UserService
from tests.test_utils.fixtures import postgres_container_fixture


@pytest.fixture
def mock_database_service(postgres_container_fixture):
    service = DatabaseService(postgres_container_fixture.get_connection_url())
    service.start()
    return service


def test_create_superuser(mock_database_service):
    # Exercise
    service = UserService(mock_database_service)
    service.create_superuser(
        "testuser",
        "test@gmail.com",
        "password123",
        github_token="gh_token"
    )

# TODO test login
