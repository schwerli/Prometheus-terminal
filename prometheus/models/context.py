from pydantic import BaseModel
from typing import Optional


class Context(BaseModel):
    """
    Context for Software Engineering tasks.
    """
    relative_path: str
    content: str
    start_line_number: Optional[int] = None
    end_line_number: Optional[int] = None

    def __str__(self):
        result = f"File: {self.relative_path}\n"
        if self.start_line_number is not None and self.end_line_number is not None:
            result += f"Line number range: {self.start_line_number} - {self.end_line_number}\n"
        result += f"Content:\n{self.content}\n"
        return result
