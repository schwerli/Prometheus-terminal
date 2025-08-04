from typing import TypeVar, Generic

from pydantic import BaseModel

T = TypeVar('T')


class Response(BaseModel, Generic[T]):
    """
    Generic response model for API responses.
    """
    code: int = 200
    message: str = "Success"
    data: T | None = None
