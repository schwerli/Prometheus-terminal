def pre_append_line_numbers(text: str, start_line: int) -> str:
  return "\n".join([f"{start_line + i}. {line}" for i, line in enumerate(text.splitlines())])
