import pytest
from sqlmodel import Session, select

from prometheus.app.db import create_superuser
from prometheus.app.entity.user import User


@pytest.mark.slow
def test_create_superuser(postgres_container_fixture, test_engine):
    username = "admin"
    email = "admin@example.com"
    password = "strongpassword"
    github_token = "ghp_123456"

    # Create superuser
    create_superuser(
        username=username,
        email=email,
        password=password,
        github_token=github_token,
    )

    with Session(test_engine) as session:
        user = session.exec(select(User).where(User.username == username)).first()

        assert user is not None
        assert user.email == email
        assert user.is_superuser is True
        assert user.issue_credit == 999999
        assert user.github_token == github_token
        assert user.password_hash != password  # Password must be hashed


@pytest.mark.slow
def test_duplicate_superuser_raises(postgres_container_fixture, test_engine):
    username = "admin2"
    email = "admin2@example.com"
    password = "password"

    create_superuser(username, email, password)

    # Trying again with same username or email
    with pytest.raises(ValueError, match="Username 'admin2' already exists"):
        create_superuser(username, "different@example.com", "pass")

    with pytest.raises(ValueError, match="Email 'admin2@example.com' already exists"):
        create_superuser("different_user", email, "pass")
