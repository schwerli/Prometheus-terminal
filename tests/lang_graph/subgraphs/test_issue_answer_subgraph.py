from prometheus.lang_graph.subgraphs.issue_answer_subgraph import IssueAnswerSubgraph
from tests.test_utils.fixtures import neo4j_container_with_kg_fixture  # noqa: F401
from tests.test_utils.util import FakeListChatWithToolsModel


def test_context_provider_subgraph(neo4j_container_with_kg_fixture):  # noqa: F811
  neo4j_container, kg = neo4j_container_with_kg_fixture
  fake_context = "Fake context"
  fake_summary = "Fake summary"
  fake_response = "Fake response"
  fake_llm = FakeListChatWithToolsModel(responses=[fake_context, fake_summary, fake_response])
  ia_subgraph = IssueAnswerSubgraph(fake_llm, kg, neo4j_container.get_driver())

  issue_response = ia_subgraph.invoke(
    "Issue title",
    "Issue body",
    [{"username": "user1", "body": "body1"}, {"username": "user2", "body": "body2"}],
  )

  assert issue_response == fake_response
