from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from prometheus.lang_graph.nodes.issue_responder_node import IssueResponderNode
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState


def test_format_verification_status_build_only_passed():
  node = IssueResponderNode(model=None)
  state = IssueAnswerAndFixState({"run_build": True, "exist_build": True, "build_fail_log": ""})
  result = node.format_verification_status(state)
  assert result == "Verification Results:\nBuild Status: PASSED"


def test_format_verification_status_build_only_failed():
  node = IssueResponderNode(model=None)
  state = IssueAnswerAndFixState(
    {"run_build": True, "exist_build": True, "build_fail_log": "Build failed due to syntax error"}
  )
  result = node.format_verification_status(state)
  assert result == "Verification Results:\nBuild Status: FAILED"


def test_format_verification_status_test_only_passed():
  node = IssueResponderNode(model=None)
  state = IssueAnswerAndFixState({"run_test": True, "exist_test": True, "test_fail_log": None})
  result = node.format_verification_status(state)
  assert result == "Verification Results:\nTest Status: PASSED"


def test_format_verification_status_test_only_failed():
  node = IssueResponderNode(model=None)
  state = IssueAnswerAndFixState(
    {"run_test": True, "exist_test": True, "test_fail_log": "Test failed in test_module.py"}
  )
  result = node.format_verification_status(state)
  assert result == "Verification Results:\nTest Status: FAILED"


def test_format_verification_status_both_passed():
  node = IssueResponderNode(model=None)
  state = IssueAnswerAndFixState(
    {
      "run_build": True,
      "exist_build": True,
      "build_fail_log": "",
      "run_test": True,
      "exist_test": True,
      "test_fail_log": "",
    }
  )
  result = node.format_verification_status(state)
  expected = "Verification Results:\nBuild Status: PASSED\nTest Status: PASSED"
  assert result == expected


def test_format_verification_status_both_failed():
  node = IssueResponderNode(model=None)
  state = IssueAnswerAndFixState(
    {
      "run_build": True,
      "exist_build": True,
      "build_fail_log": "Build failed",
      "run_test": True,
      "exist_test": True,
      "test_fail_log": "Tests failed",
    }
  )
  result = node.format_verification_status(state)
  expected = "Verification Results:\nBuild Status: FAILED\nTest Status: FAILED"
  assert result == expected


def test_format_verification_status_run_but_not_exist():
  node = IssueResponderNode(model=None)
  state = IssueAnswerAndFixState(
    {"run_build": True, "exist_build": False, "run_test": True, "exist_test": False}
  )
  result = node.format_verification_status(state)
  assert result == ""


def test_format_human_message_basic():
  node = IssueResponderNode(model=None)
  state = IssueAnswerAndFixState(
    {
      "issue_title": "Test Issue",
      "issue_body": "This is a test issue body",
      "summary": "Test summary of the issue",
    }
  )
  result = node.format_human_message(state)

  assert isinstance(result, HumanMessage)
  assert "MODE: Information Only (No Code Changes)" in result.content
  assert "Test Issue" in result.content
  assert "This is a test issue body" in result.content
  assert "Test summary of the issue" in result.content
  assert "CHANGES MADE:" not in result.content
  assert "Verification Results:" not in result.content


def test_format_human_message_with_patch():
  node = IssueResponderNode(model=None)
  state_with_patch = IssueAnswerAndFixState(
    {
      "issue_title": "Test Issue",
      "issue_body": "This is a test issue body",
      "summary": "Test summary of the issue",
      "patch": "diff --git a/test.py b/test.py\n+new line",
    }
  )
  result = node.format_human_message(state_with_patch)

  assert isinstance(result, HumanMessage)
  assert "MODE: Solution With Code Changes" in result.content
  assert "CHANGES MADE:" in result.content
  assert "diff --git a/test.py b/test.py" in result.content
  assert "Verification Results:" not in result.content


def test_format_human_message_with_verification():
  node = IssueResponderNode(model=None)
  state_with_verification = IssueAnswerAndFixState(
    {
      "issue_title": "Test Issue",
      "issue_body": "This is a test issue body",
      "summary": "Test summary of the issue",
      "patch": "diff --git a/test.py b/test.py\n+new line",
      "run_build": True,
      "exist_build": True,
      "build_fail_log": "",
      "run_test": True,
      "exist_test": True,
      "test_fail_log": "",
    }
  )
  result = node.format_human_message(state_with_verification)

  assert isinstance(result, HumanMessage)
  assert "MODE: Solution With Code Changes" in result.content
  assert "CHANGES MADE:" in result.content
  assert "Verification Results:" in result.content
  assert "Build Status: PASSED" in result.content
  assert "Test Status: PASSED" in result.content


def test_call_basic_functionality():
  fake_response = "Fake response content"
  fake_model = FakeListChatModel(responses=[fake_response])
  node = IssueResponderNode(model=fake_model)
  basic_state = IssueAnswerAndFixState(
    {
      "issue_title": "Test Issue",
      "issue_body": "This is a test issue body",
      "summary": "Test summary of the issue",
    }
  )
  result = node(basic_state)

  # Verify result structure
  assert isinstance(result, dict)
  assert "issue_response" in result
  assert result["issue_response"] == fake_response
