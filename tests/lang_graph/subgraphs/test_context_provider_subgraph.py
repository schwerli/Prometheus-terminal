import pytest

from prometheus.lang_graph.subgraphs.context_provider_subgraph import ContextProviderSubgraph
from tests.test_utils.fixtures import neo4j_container_with_kg_fixture  # noqa: F401
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.mark.slow
def test_context_provider_subgraph(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, kg = neo4j_container_with_kg_fixture
  fake_context = "Fake context"
  fake_summary = "Fake summary"
  fake_llm = FakeListChatWithToolsModel(responses=[fake_context, fake_summary])
  cp_subgraph = ContextProviderSubgraph(fake_llm, kg, neo4j_container.get_driver(), 1000)

  summary = cp_subgraph.invoke("Dummy query")

  assert summary == fake_summary
