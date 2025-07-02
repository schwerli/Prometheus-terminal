import pytest
from sqlmodel import create_engine

from prometheus.app.db import create_db_and_tables


@pytest.fixture(scope="session")
def test_engine(postgres_container_fixture):
    url = postgres_container_fixture.get_connection_url()
    engine = create_engine(url, echo=True)
    create_db_and_tables()
    yield engine
