from langchain_core.messages import HumanMessage

from prometheus.lang_graph.nodes.general_build_node import GeneralBuildNode
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_state import IssueAnswerAndFixState
from tests.test_utils.util import FakeListChatWithToolsModel


def test_format_human_message_before_edit():
  fake_model = FakeListChatWithToolsModel(responses=[])
  build_node = GeneralBuildNode(fake_model, None, before_edit=True)
  state: IssueAnswerAndFixState = {
    "project_structure": "sample/project/structure",
    "build_summary": "previous build summary",
  }

  result = build_node.format_human_message(state)

  assert isinstance(result, HumanMessage)
  assert result.content == "The (incomplete) project structure is:\nsample/project/structure"
  assert "build_summary" not in result.content


def test_format_human_message_after_edit():
  fake_model = FakeListChatWithToolsModel(responses=[])
  build_node = GeneralBuildNode(fake_model, None, before_edit=False)
  state: IssueAnswerAndFixState = {
    "project_structure": "sample/project/structure",
    "build_summary": "previous build summary",
  }

  result = build_node.format_human_message(state)

  assert isinstance(result, HumanMessage)
  assert result.content == (
    "The (incomplete) project structure is:\nsample/project/structure\n\n"
    "The previous build summary is:\nprevious build summary"
  )


def test_call_before_edit():
  # Setup fake model with a predefined response
  fake_response = "Build analysis complete"
  fake_model = FakeListChatWithToolsModel(responses=[fake_response])
  build_node = GeneralBuildNode(fake_model, None, before_edit=True)

  state: IssueAnswerAndFixState = {
    "project_structure": "sample/project/structure",
    "build_messages": [],
  }

  result = build_node(state)

  assert "build_messages" in result
  assert len(result["build_messages"]) == 1
  assert result["build_messages"][0].content == fake_response


def test_call_after_edit_no_build():
  fake_model = FakeListChatWithToolsModel(responses=[])
  build_node = GeneralBuildNode(fake_model, None, before_edit=False)

  state: IssueAnswerAndFixState = {
    "project_structure": "sample/project/structure",
    "build_messages": [],
    "exist_build": False,
  }

  result = build_node(state)

  assert "build_messages" in result
  assert len(result["build_messages"]) == 1
  assert (
    result["build_messages"][0].content == "Previous agent determined there is no build system."
  )


def test_call_after_edit_with_build():
  fake_response = "Build analysis after edit"
  fake_model = FakeListChatWithToolsModel(responses=[fake_response])
  build_node = GeneralBuildNode(fake_model, None, before_edit=False)

  state: IssueAnswerAndFixState = {
    "project_structure": "sample/project/structure",
    "build_summary": "previous build summary",
    "build_messages": [],
    "exist_build": True,
  }

  result = build_node(state)

  assert "build_messages" in result
  assert len(result["build_messages"]) == 1
  assert result["build_messages"][0].content == fake_response
