



from unittest.mock import Mock

import neo4j
import pytest

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.issue_answer_and_fix_subgraph import IssueAnswerAndFixSubgraph
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_kg():
    return Mock(spec=KnowledgeGraph)

@pytest.fixture
def mock_neo4j_driver():
    return Mock(spec=neo4j.Driver)

@pytest.fixture
def temp_project_dir(tmp_path):
    # Create a temporary project structure
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "src").mkdir()
    (project_dir / "tests").mkdir()
    return project_dir


def test_issue_answer_fix_subgraph_initialization(
    mock_kg,
    mock_neo4j_driver,
    temp_project_dir
):
    """Test that IssueAnswerAndFixSubgraph initializes correctly with all components."""
    
    fake_model = FakeListChatWithToolsModel(responses=[])

    # Initialize the subgraph
    subgraph = IssueAnswerAndFixSubgraph(
        model=fake_model,
        kg=mock_kg,
        neo4j_driver=mock_neo4j_driver,
        local_path=temp_project_dir
    )
    
    # Verify the subgraph was created
    assert subgraph.subgraph is not None
    assert subgraph.local_path == temp_project_dir.absolute()