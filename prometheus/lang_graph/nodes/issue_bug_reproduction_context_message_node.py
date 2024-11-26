import logging

from langchain_core.messages import HumanMessage

from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.utils.issue_util import format_issue_info


class IssueBugReproductionContextMessageNode:
  HUMAN_PROMPT = """\
{issue_info}

OBJECTIVE: Find three relevant existing test cases that demonstrates similar functionality to the reported bug,
including the test setup, mocking, and assertions.

<reasoning>
1. Analyze bug characteristics:
   - Core functionality being tested
   - Input parameters and configurations
   - Expected error conditions
   - Environmental dependencies

2. Search requirements:
   - Test files exercising similar functionality
   - Mock/fixture setup patterns
   - Assertion styles
   - Error handling tests

3. Focus areas:
   - Test setup and teardown
   - Mock object configuration
   - Network/external service simulation
   - Error condition verification
</reasoning>

REQUIREMENTS:
- Return ONE complete, self-contained test case most similar to bug scenario
- Must include full test method implementation
- Must include ALL mock/fixture setup
- Must include helper functions used by test
- Must preserve exact file paths and line numbers

<examples>
<example id="database-timeout">
<bug>
db.execute("SELECT * FROM users").fetchall() 
raises ConnectionTimeout when load is high
</bug>

<ideal_test_match>
# File: tests/test_database.py
class TestDatabaseTimeout:
    @pytest.fixture
    def mock_db_connection(self):
        conn = Mock()
        conn.execute.side_effect = [
            ConnectionTimeout("Connection timed out"),
            QueryResult(["user1", "user2"])  # Second try succeeds
        ]
        return conn
        
    def test_handle_timeout_during_query(self, mock_db_connection):
        # Complete test showing timeout scenario
        # Including retry logic verification
        # With all necessary assertions
</ideal_test_match>
</example>

<example id="file-permission">
<bug>
FileProcessor('/root/data.txt').process() 
fails with PermissionError
</bug>

<ideal_test_match>
# File: tests/test_file_processor.py
class TestFilePermissions:
    @patch('os.access')
    @patch('builtins.open')
    def test_file_permission_denied(self, mock_open, mock_access):
        # Full test setup with mocked file system
        # Permission denial simulation
        # Error handling verification
</ideal_test_match>
</example>

Search priority:
1. Tests of exact same functionality
2. Tests with similar error conditions
3. Tests with comparable mocking patterns
4. Tests demonstrating similar assertions

Return the THREE most relevant test cases with complete context.
"""

  def __init__(self):
    self._logger = logging.getLogger(
      "prometheus.lang_graph.nodes.issue_bug_reproduction_context_message_node"
    )

  def __call__(self, state: BugReproductionState):
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(
        issue_info=format_issue_info(
          state["issue_title"], state["issue_body"], state["issue_comments"]
        ),
      )
    )
    self._logger.debug(f"Sending query to context provider:\n{human_message}")
    return {"context_provider_messages": [human_message]}
