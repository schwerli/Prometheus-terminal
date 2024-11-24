from unittest.mock import Mock

import neo4j
import pytest

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.context_provider_subgraph import ContextProviderSubgraph
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_kg():
  kg = Mock(spec=KnowledgeGraph)
  kg.get_all_ast_node_types.return_value = ["function_def"]
  return kg


@pytest.fixture
def mock_neo4j_driver():
  return Mock(spec=neo4j.Driver)


def test_context_provider_subgraph_basic_initialization(mock_kg, mock_neo4j_driver):
  """Test that ContextProviderSubgraph initializes correctly with basic components."""
  # Initialize fake model with empty responses
  fake_model = FakeListChatWithToolsModel(responses=[])

  # Initialize the subgraph with required parameters
  subgraph = ContextProviderSubgraph(
    model=fake_model, kg=mock_kg, neo4j_driver=mock_neo4j_driver, max_token_per_neo4j_result=1000
  )

  # Verify the subgraph was created
  assert subgraph.subgraph is not None
  assert subgraph.thread_id is None
