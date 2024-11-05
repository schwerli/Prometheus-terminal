from unittest.mock import MagicMock, Mock, call, create_autospec

import neo4j
import pytest

from prometheus.utils.neo4j_util import format_neo4j_result, run_neo4j_query


class MockResult:
    def __init__(self, data_list):
        self._data = data_list
        
    def data(self):
        return self._data


class MockSession:
    def __init__(self):
        self.execute_read = MagicMock()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockDriver:
    def __init__(self, session):
        self._session = session
    
    def session(self):
        return self._session


@pytest.fixture
def mock_neo4j_driver():
    session = MockSession()
    driver = MockDriver(session)
    return driver, session



def test_format_neo4j_result_single_row():
    result = MockResult([
        {"name": "John", "age": 30}
    ])
    
    formatted = format_neo4j_result(result)
    expected = "Result 1:\nage: 30\nname: John"
    
    assert formatted == expected


def test_format_neo4j_result_multiple_rows():
    result = MockResult([
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25}
    ])

    formatted = format_neo4j_result(result)
    expected = "Result 1:\nage: 30\nname: John\n\n\nResult 2:\nage: 25\nname: Jane"

    assert formatted == expected


def test_format_neo4j_result_empty():
    result = MockResult([])
    formatted = format_neo4j_result(result)
    assert formatted == ""


def test_format_neo4j_result_different_keys():
    result = MockResult([
        {"name": "John", "age": 30},
        {"city": "New York", "country": "USA"}
    ])
    
    formatted = format_neo4j_result(result)
    expected = "Result 1:\nage: 30\nname: John\n\n\nResult 2:\ncity: New York\ncountry: USA"
    
    assert formatted == expected


def test_format_neo4j_result_complex_values():
    result = MockResult([
        {
            "numbers": [1, 2, 3],
            "metadata": {"type": "user", "active": True},
            "date": "2024-01-01"
        }
    ])
    
    formatted = format_neo4j_result(result)
    expected = "Result 1:\ndate: 2024-01-01\nmetadata: {'type': 'user', 'active': True}\nnumbers: [1, 2, 3]"
    
    assert formatted == expected


def test_run_neo4j_query_success(mock_neo4j_driver):
    driver, session = mock_neo4j_driver
    
    # Create the expected formatted result
    test_data = [{"name": "John", "age": 30}]
    expected = "Result 1:\nage: 30\nname: John"
    
    # Mock the transaction execution to return the formatted result
    def execute_read_side_effect(query_func):
        # Create a mock transaction
        tx = Mock()
        # Set up the mock result
        tx.run.return_value = MockResult(test_data)
        # Execute the query function with our mock transaction
        return query_func(tx)
    
    # Set up the session mock to use our side effect
    session.execute_read.side_effect = execute_read_side_effect
    
    # Run the query
    query = "MATCH (n:Person) RETURN n.name as name, n.age as age"
    result = run_neo4j_query(query, driver)
    
    # Verify results
    assert result == expected
    
    # Verify the session was used correctly
    session.execute_read.assert_called_once()


def test_run_neo4j_query_empty_result(mock_neo4j_driver):
    driver, session = mock_neo4j_driver
    
    def execute_read_side_effect(func):
        mock_tx = Mock()
        mock_tx.run.return_value = MockResult([])
        return func(mock_tx)
    
    session.execute_read.side_effect = execute_read_side_effect
    
    result = run_neo4j_query("MATCH (n:NonExistent) RETURN n", driver)
    assert result == ""


def test_run_neo4j_query_multiple_results(mock_neo4j_driver):
    driver, session = mock_neo4j_driver
    
    def execute_read_side_effect(func):
        mock_tx = Mock()
        mock_tx.run.return_value = MockResult([
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25}
        ])
        return func(mock_tx)
    
    session.execute_read.side_effect = execute_read_side_effect
    
    result = run_neo4j_query("MATCH (n:Person) RETURN n.name as name, n.age as age", driver)
    expected = "Result 1:\nage: 30\nname: John\n\n\nResult 2:\nage: 25\nname: Jane"
    assert result == expected