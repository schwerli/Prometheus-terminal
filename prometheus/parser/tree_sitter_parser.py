from pathlib import Path

from tree_sitter._binding import Tree
from tree_sitter_languages import get_parser

from prometheus.parser.file_types import FileType


class FileNotSupportedError(Exception):
  pass


FILE_TYPE_TO_LANG = {
  FileType.BASH: "bash",
  FileType.C: "c",
  FileType.CSHARP: "c_sharp",
  FileType.CPP: "cpp",
  FileType.GO: "go",
  FileType.JAVA: "java",
  FileType.JAVASCRIPT: "javascript",
  FileType.KOTLIN: "kotlin",
  FileType.PHP: "php",
  FileType.PYTHON: "python",
  FileType.SQL: "sql",
  FileType.YAML: "yaml",
}


def supports_file(file: Path) -> bool:
  file_type = FileType.from_path(file)
  return file_type in FILE_TYPE_TO_LANG


def parse(file: Path) -> Tree:
  file_type = FileType.from_path(file)
  lang = FILE_TYPE_TO_LANG.get(file_type, None)
  if lang is None:
    raise FileNotSupportedError(
      f"{file_type.value} is not supported by tree_sitter_parser"
    )

  lang_parser = get_parser(lang)
  with file.open("rb") as f:
    return lang_parser.parse(f.read())
