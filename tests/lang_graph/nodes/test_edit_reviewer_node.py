from langchain_core.messages import HumanMessage

from prometheus.lang_graph.nodes.edit_reviewer_node import EditReviewerNode
from tests.test_utils.util import FakeListChatWithToolsModel


def test_edit_reviewer_node_initialization():
  """Test initialization of EditReviewerNode"""
  fake_model = FakeListChatWithToolsModel(responses=[])
  root_path = "/test/path"

  edit_reviewer_node = EditReviewerNode(fake_model, root_path)

  assert edit_reviewer_node.model_with_tool is not None
  assert len(edit_reviewer_node.tools) > 0
  assert edit_reviewer_node.system_prompt.content is not None


def test_edit_reviewer_node_format_issue_comments():
  """Test formatting of issue comments"""
  fake_model = FakeListChatWithToolsModel(responses=[])
  root_path = "/test/path"

  edit_reviewer_node = EditReviewerNode(fake_model, root_path)

  issue_comments = [
    {"username": "user1", "comment": "First comment"},
    {"username": "user2", "comment": "Second comment"},
  ]

  formatted_comments = edit_reviewer_node.format_issue_comments(issue_comments)

  assert "user1: First comment" in formatted_comments
  assert "user2: Second comment" in formatted_comments


def test_edit_reviewer_node_format_human_message():
  """Test formatting human message for the language model"""
  fake_model = FakeListChatWithToolsModel(responses=[])
  root_path = "/test/path"

  edit_reviewer_node = EditReviewerNode(fake_model, root_path)

  state = {
    "issue_title": "Test Issue",
    "issue_body": "Test Description",
    "issue_comments": [{"username": "user", "comment": "Test comment"}],
    "summary": "Test context summary",
    "patch": "Test patch",
    "edit_reviewer_messages": [],
  }

  human_message = edit_reviewer_node.format_human_message(state)

  assert isinstance(human_message, HumanMessage)
  assert "Test Issue" in human_message.content
  assert "Test Description" in human_message.content
  assert "user: Test comment" in human_message.content
  assert "Test context summary" in human_message.content
  assert "Test patch" in human_message.content


def test_edit_reviewer_node_call_method():
  """Test the __call__ method of EditReviewerNode"""
  # Simulate a model response with a string instead of AIMessage
  mock_response = "Test Review Response"
  fake_model = FakeListChatWithToolsModel(responses=[mock_response])
  root_path = "/test/path"

  edit_reviewer_node = EditReviewerNode(fake_model, root_path)

  state = {
    "issue_title": "Test Issue",
    "issue_body": "Test Description",
    "issue_comments": [{"username": "user", "comment": "Test comment"}],
    "summary": "Test context summary",
    "patch": "Test patch",
    "edit_reviewer_messages": [],
  }

  result = edit_reviewer_node(state)

  assert "edit_reviewer_messages" in result
  assert len(result["edit_reviewer_messages"]) == 1
  assert result["edit_reviewer_messages"][0].content == mock_response
