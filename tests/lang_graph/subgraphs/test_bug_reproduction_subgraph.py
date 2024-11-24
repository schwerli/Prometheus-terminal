from unittest.mock import Mock

import pytest

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.bug_reproduction_subgraph import BugReproductionSubgraph
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_container():
  return Mock(spec=BaseContainer)


@pytest.fixture
def mock_kg():
  kg = Mock(spec=KnowledgeGraph)
  kg.get_local_path.return_value = "/foo/bar"
  return kg


def test_bug_reproduction_subgraph_basic_initialization(mock_container, mock_kg):
  """Test that BugReproductionSubgraph initializes correctly with basic components."""
  # Initialize fake model with empty responses
  fake_model = FakeListChatWithToolsModel(responses=[])

  # Initialize the subgraph
  subgraph = BugReproductionSubgraph(fake_model, mock_container, mock_kg)

  # Verify the subgraph was created
  assert subgraph.subgraph is not None
  assert subgraph.thread_id is None
