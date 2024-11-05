import os
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.utils.str_util import pre_append_line_numbers


class ReadFileInput(BaseModel):
  relative_path: str = Field("The relative path of the file to read")


READ_FILE_DESCRIPTION = """\
Read the content of a file from the codebase with a safety limit on the number of lines.
Returns up to the first 1000 lines by default to prevent context issues with large files.
Returns an error message if the file doesn't exist.
"""


def read_file(relative_path: str, root_path: str, n_lines: int = 1000) -> str:
  file_path = Path(os.path.join(root_path, relative_path))
  if not file_path.exists():
    return f"The file {relative_path} does not exist."

  with file_path.open() as f:  # newline='' preserves original line endings
    lines = f.readlines()

  return "".join(lines[:n_lines])


class ReadFileWithLineNumbersInput(BaseModel):
  relative_path: str = Field(description="The relative path of the file to read")
  start_line: int = Field(description="The start line number to read, 1-indexed and inclusive")
  end_line: int = Field(description="The ending line number to read, 1-indexed and inclusive")


READ_FILE_WITH_LINE_NUMBERS_DESCRIPTION = """\
Read a specific range of lines from a file and return the content with line numbers prepended.
The line numbers are 1-indexed and both start and end lines are inclusive.
"""


def read_file_with_line_numbers(
  relative_path: str, root_path: str, start_line: int, end_line: int
) -> str:
  file_path = Path(os.path.join(root_path, relative_path))
  if not file_path.exists():
    return f"The file {relative_path} does not exist."

  if end_line < start_line:
    return f"The end line number {end_line} is less than the start line number {start_line}."

  zero_based_start_line = start_line - 1
  zero_based_end_line = end_line

  with file_path.open() as f:
    lines = f.readlines()

  return pre_append_line_numbers(
    "".join(lines[zero_based_start_line:zero_based_end_line]), start_line
  )


class CreateFileInput(BaseModel):
  relative_path: str = Field(description="The relative path of the file to create")
  content: str = Field(description="The content of the file to create")


CREATE_FILE_DESCRIPTION = """\
Create a new file at the specified path with the given content. 
If the parent directories don't exist, they will be created automatically.
Returns an error message if the file already exists.
"""


def create_file(relative_path: str, root_path: str, content: str) -> str:
  file_path = Path(os.path.join(root_path, relative_path))
  if file_path.exists():
    return f"The file {relative_path} already exists."

  file_path.parent.mkdir(parents=True, exist_ok=True)
  file_path.write_text(content)
  return f"The file {relative_path} has been created."


class DeleteInput(BaseModel):
  relative_path: str = Field(description="The relative path of the file/dir to delete")


DELETE_DESCRIPTION = """\
Delete a file or directory at the specified path.
For directories, it will recursively delete all contents.
Returns an error message if the path doesn't exist.
"""


def delete(relative_path: str, root_path: str) -> str:
  file_path = Path(os.path.join(root_path, relative_path))
  if not file_path.exists():
    return f"The file {relative_path} does not exist."

  if file_path.is_dir():
    shutil.rmtree(file_path)
    return f"The directory {relative_path} has been deleted."

  file_path.unlink()
  return f"The file {relative_path} has been deleted."


class EditFileInput(BaseModel):
  relative_path: str = Field(description="The relative path of the file to edit")
  start_line: int = Field(description="The start line number to edit, 1-indexed and inclusive")
  end_line: int = Field(description="The ending line number to edit, 1-indexed and inclusive")
  new_content: str = Field(
    description="The new content to write to the file between start_line and end_line"
  )


EDIT_FILE_DESCRIPTION = """\
Edit a specific range of lines in an existing file.
Replaces the content between start_line and end_line (inclusive, 1-indexed) with the new content.
Automatically adds a newline to the new content if it doesn't end with one.
Returns an error message if the file doesn't exist or if end_line is less than start_line.
"""


def edit_file(
  relative_path: str, root_path: str, start_line: int, end_line: int, new_content: str
) -> str:
  file_path = Path(os.path.join(root_path, relative_path))
  if not file_path.exists():
    return f"The file {relative_path} does not exist."

  if end_line < start_line:
    return f"The end line number {end_line} is less than the start line number {start_line}."

  zero_based_start_line = start_line - 1
  zero_based_end_line = end_line

  with file_path.open() as f:
    lines = f.readlines()

  if not new_content.endswith("\n"):
    new_content += "\n"

  lines[zero_based_start_line:zero_based_end_line] = new_content.splitlines(True)
  file_path.write_text("".join(lines))
  return f"The file {relative_path} has been edited."
