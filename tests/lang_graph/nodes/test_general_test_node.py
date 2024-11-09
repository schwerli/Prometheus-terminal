from langchain_core.messages import HumanMessage

from prometheus.lang_graph.nodes.general_test_node import GeneralTestNode
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from tests.test_utils.util import FakeListChatWithToolsModel


def test_format_human_message_before_edit():
  fake_model = FakeListChatWithToolsModel(responses=[])
  test_node = GeneralTestNode(fake_model, None, before_edit=True)
  state: IssueAnswerAndFixState = {
    "project_structure": "sample/project/structure",
    "test_summary": "previous test summary",
  }

  result = test_node.format_human_message(state)

  assert isinstance(result, HumanMessage)
  assert result.content == "The (incomplete) project structure is:\nsample/project/structure"
  assert "test_summary" not in result.content


def test_format_human_message_after_edit():
  fake_model = FakeListChatWithToolsModel(responses=[])
  test_node = GeneralTestNode(fake_model, None, before_edit=False)
  state: IssueAnswerAndFixState = {
    "project_structure": "sample/project/structure",
    "test_command_summary": "previous test summary",
  }

  result = test_node.format_human_message(state)

  assert isinstance(result, HumanMessage)
  assert result.content == (
    "The (incomplete) project structure is:\nsample/project/structure\n\n"
    "The previous test summary is:\nprevious test summary"
  )


def test_call_before_edit():
  # Setup fake model with a predefined response
  fake_response = "Test analysis complete"
  fake_model = FakeListChatWithToolsModel(responses=[fake_response])
  test_node = GeneralTestNode(fake_model, None, before_edit=True)

  state: IssueAnswerAndFixState = {
    "project_structure": "sample/project/structure",
    "test_messages": [],
  }

  result = test_node(state)

  assert "test_messages" in result
  assert len(result["test_messages"]) == 1
  assert result["test_messages"][0].content == fake_response


def test_call_after_edit_no_test():
  fake_model = FakeListChatWithToolsModel(responses=[])
  test_node = GeneralTestNode(fake_model, None, before_edit=False)

  state: IssueAnswerAndFixState = {
    "project_structure": "sample/project/structure",
    "test_messages": [],
    "exist_test": False,
  }

  result = test_node(state)

  assert "build_messages" in result
  assert len(result["build_messages"]) == 1
  assert (
    result["build_messages"][0].content == "Previous agent determined there is no test framework."
  )


def test_call_after_edit_with_test():
  fake_response = "Test analysis after edit"
  fake_model = FakeListChatWithToolsModel(responses=[fake_response])
  test_node = GeneralTestNode(fake_model, None, before_edit=False)

  state: IssueAnswerAndFixState = {
    "project_structure": "sample/project/structure",
    "test_command_summary": "previous test summary",
    "test_messages": [],
    "exist_test": True,
  }

  result = test_node(state)

  assert "test_messages" in result
  assert len(result["test_messages"]) == 1
  assert result["test_messages"][0].content == fake_response
