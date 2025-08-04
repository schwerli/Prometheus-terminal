import logging
from typing import Optional

from passlib.hash import bcrypt
from sqlmodel import Session

from prometheus.app.entity.user import User
from prometheus.app.services.base_service import BaseService
from prometheus.app.services.database_service import DatabaseService


class UserService(BaseService):
    def __init__(self, database_service: DatabaseService):
        self.database_service = database_service
        self.engine = database_service.engine
        self._logger = logging.getLogger("prometheus.app.services.user_service")

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        github_token: Optional[str] = None,
        issue_credit: int = 0,
        is_superuser: bool = False,
    ) -> None:
        """
        Create a new superuser and commit it to the database.

        Args:
            username (str): Desired username.
            email (str): Email address.
            password (str): Plaintext password (will be hashed).
            github_token (Optional[str]): Optional GitHub token.
            issue_credit (int): Optional issue credit.
            is_superuser (bool): Whether the user is a superuser.
        Returns:
            User: The created superuser instance.
        """
        with Session(self.engine) as session:
            if session.query(User).filter(User.username == username).first():
                raise ValueError(f"Username '{username}' already exists")
            if session.query(User).filter(User.email == email).first():
                raise ValueError(f"Email '{email}' already exists")

            hashed_password = bcrypt.hash(password)

            user = User(
                username=username,
                email=email,
                password_hash=hashed_password,
                github_token=github_token,
                issue_credit=issue_credit,
                is_superuser=is_superuser,
            )
            session.add(user)
            session.commit()
            session.refresh(user)

    # Create a superuser and commit it to the database
    def create_superuser(
        self,
        username: str,
        email: str,
        password: str,
        github_token: Optional[str] = None,
    ) -> None:
        """
        Create a new superuser in the database.

        This method creates a superuser with the provided credentials and commits it to the database.
        """
        self.create_user(
            username, email, password, github_token, is_superuser=True, issue_credit=999999
        )
        self._logger.info(f"Superuser '{username}' created successfully.")
