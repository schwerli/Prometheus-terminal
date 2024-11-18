from prometheus.utils.str_util import (
  TRUNCATED_TEXT,
  get_tokenizer,
  pre_append_line_numbers,
  truncate_text,
)


def test_single_line():
  text = "Hello world"
  result = pre_append_line_numbers(text, start_line=1)
  assert result == "1. Hello world"


def test_multiple_lines():
  text = "First line\nSecond line\nThird line"
  result = pre_append_line_numbers(text, start_line=1)
  assert result == "1. First line\n2. Second line\n3. Third line"


def test_empty_string():
  text = ""
  result = pre_append_line_numbers(text, start_line=1)
  assert result == ""


def test_no_truncation_needed():
  text = "Short text"
  result = truncate_text(text, max_token=100)
  assert result == text


def test_truncation():
  # Generate text that will definitely need truncation
  long_text = "Hello world! " * 50
  max_tokens = 20
  result = truncate_text(long_text, max_tokens)

  # Verify length
  assert len(get_tokenizer().encode(result)) <= max_tokens
  # Verify warning message
  assert result.endswith(TRUNCATED_TEXT)
  # Verify some original content remains
  assert result.startswith("Hello world!")


def test_truncation_with_empty_string():
  result = truncate_text("", max_token=10)
  assert result == ""


def test_truncation_with_unicode_handling():
  text = "Hello 👋 World 🌍"
  result = truncate_text(text, max_token=100)
  assert result == text
