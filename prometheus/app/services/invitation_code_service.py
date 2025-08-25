import logging
import uuid
from typing import Sequence

from sqlmodel import Session, select

from prometheus.app.entity.invitation_code import InvitationCode
from prometheus.app.services.base_service import BaseService
from prometheus.app.services.database_service import DatabaseService


class InvitationCodeService(BaseService):
    def __init__(self, database_service: DatabaseService):
        self.database_service = database_service
        self.engine = database_service.engine
        self._logger = logging.getLogger("prometheus.app.services.invitation_code_service")

    def create_invitation_code(self) -> InvitationCode:
        """
        Create a new invitation code and commit it to the database.

        Returns:
            InvitationCode: The created invitation code instance.
        """

        with Session(self.engine) as session:
            code = str(uuid.uuid4())
            invitation_code = InvitationCode(code=code)
            session.add(invitation_code)
            session.commit()
            session.refresh(invitation_code)
            return invitation_code

    def list_invitation_codes(self) -> Sequence[InvitationCode]:
        """
        List all invitation codes from the database.

        Returns:
            Sequence[InvitationCode]: A list of all invitation code instances.
        """
        with Session(self.engine) as session:
            statement = select(InvitationCode)
            return session.exec(statement).all()
