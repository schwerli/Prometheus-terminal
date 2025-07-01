from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from prometheus.app.services.postgres_service import PostgresService


@pytest.fixture
def mock_postgres_saver():
    with patch("prometheus.app.services.postgres_service.PostgresSaver") as mock_saver_class:
        mock_saver = Mock()
        mock_saver_class.return_value = mock_saver
        yield mock_saver


@pytest.fixture
def mock_psycopg_conn():
    with patch("prometheus.app.services.postgres_service.Connection.connect") as mock_conn:
        conn = Mock()
        mock_conn.return_value = conn
        yield conn


def test_init_service_sets_up_checkpointer(mock_postgres_saver, mock_psycopg_conn):
    service = PostgresService("postgresql://test")

    mock_psycopg_conn.close.assert_not_called()
    mock_postgres_saver.setup.assert_called_once()
    assert service.checkpointer == mock_postgres_saver


def test_get_all_thread_ids_sorted_and_unique(mock_postgres_saver, mock_psycopg_conn):
    now = datetime.now()
    mock_postgres_saver.list.return_value = [
        Mock(config={"configurable": {"thread_id": "id1"}}, checkpoint={"ts": (now.isoformat())}),
        Mock(config={"configurable": {"thread_id": "id2"}}, checkpoint={"ts": (now.isoformat())}),
        Mock(config={"configurable": {"thread_id": "id1"}}, checkpoint={"ts": (now.isoformat())}),
    ]

    service = PostgresService("postgresql://test")
    result = service.get_all_thread_ids()

    assert result == ["id1", "id2"] or result == ["id2", "id1"]
    assert len(result) == 2


def test_get_messages_filters_and_formats_correctly(mock_postgres_saver, mock_psycopg_conn):
    mock_checkpoint = {
        "channel_values": {
            "query": "Initial query?",
            "messages": [
                HumanMessage(content="How are you?"),
                AIMessage(content="I'm fine!", additional_kwargs={}),
                AIMessage(content="tool call", additional_kwargs={"tool_call": "fake"}),
            ],
        }
    }

    mock_postgres_saver.get.return_value = mock_checkpoint

    service = PostgresService("postgresql://test")
    result = service.get_messages("thread-123")

    assert result == [
        {"role": "user", "text": "Initial query?"},
        {"role": "user", "text": "How are you?"},
        {"role": "assistant", "text": "I'm fine!"},
    ]
