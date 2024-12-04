from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.bug_reproducing_write_node import BugReproducingWriteNode
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_kg():
  kg = Mock(spec=KnowledgeGraph)
  kg.get_local_path.return_value = "/foo/bar"
  return kg


@pytest.fixture
def test_state():
  return BugReproductionState(
    {
      "issue_title": "Test Bug",
      "issue_body": "Bug description",
      "issue_comments": ["Comment 1", "Comment 2"],
      "bug_context": "Context of the bug",
      "reproduced_bug_file": "test/file.py",
      "reproduced_bug_failure_log": "Test failure log",
      "bug_reproducing_write_messages": [HumanMessage("assert x == 10")],
    }
  )


def test_call_method(mock_kg, test_state):
  """Test the __call__ method execution."""
  fake_response = "Created test file"
  fake_llm = FakeListChatWithToolsModel(responses=[fake_response])
  node = BugReproducingWriteNode(fake_llm, mock_kg)

  result = node(test_state)

  assert "bug_reproducing_write_messages" in result
  assert len(result["bug_reproducing_write_messages"]) == 1
  assert result["bug_reproducing_write_messages"][0].content == fake_response
