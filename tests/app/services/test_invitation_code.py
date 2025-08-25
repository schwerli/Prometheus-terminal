import pytest
from sqlmodel import SQLModel, Session

from prometheus.app.services.invitation_code_service import InvitationCodeService
from prometheus.app.services.database_service import DatabaseService
from prometheus.app.entity.invitation_code import InvitationCode
from tests.test_utils.fixtures import postgres_container_fixture  # noqa: F401


@pytest.fixture
def mock_database_service(postgres_container_fixture):  # noqa: F811
    """Fixture: provide a clean DatabaseService using the Postgres test container."""
    service = DatabaseService(postgres_container_fixture.get_connection_url())
    service.start()
    # Initialize schema
    SQLModel.metadata.create_all(service.engine)
    yield service
    service.close()


@pytest.fixture
def service(mock_database_service):
    """Fixture: construct an InvitationCodeService with the database service."""
    return InvitationCodeService(database_service=mock_database_service)


def test_create_invitation_code(service):
    """Test that create_invitation_code correctly generates and returns an InvitationCode."""
    invitation_code = service.create_invitation_code()

    # Verify the returned object is an InvitationCode instance
    assert isinstance(invitation_code, InvitationCode)
    assert isinstance(invitation_code.code, str)
    assert len(invitation_code.code) == 36  # uuid4 string length
    assert invitation_code.id is not None

    # Verify the object is persisted in the database
    with Session(service.engine) as session:
        db_obj = session.get(InvitationCode, invitation_code.id)
        assert db_obj is not None
        assert db_obj.code == invitation_code.code


def test_list_invitation_codes(service):
    """Test that list_invitation_codes returns all stored invitation codes."""
    # Insert two invitation codes first
    code1 = service.create_invitation_code()
    code2 = service.create_invitation_code()

    codes = service.list_invitation_codes()

    # Verify length
    assert len(codes) >= 2
    # Verify both created codes are included
    all_codes = [c.code for c in codes]
    assert code1.code in all_codes
    assert code2.code in all_codes
