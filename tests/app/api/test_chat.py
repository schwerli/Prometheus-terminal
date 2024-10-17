from unittest.mock import Mock
from fastapi import FastAPI
from fastapi.testclient import TestClient
import neo4j

from langchain_community.chat_models.fake import FakeListChatModel
from prometheus.agents.context_provider_agent import ContextProviderAgent
from prometheus.app.api import chat
from prometheus.graph.knowledge_graph import KnowledgeGraph

app = FastAPI()
app.include_router(chat.router, prefix="/chat", tags=["chat"])
client = TestClient(app)


class FakeListChatModelWithTools(FakeListChatModel):
  def bind_tools(self, tools):
    return self


def test_send_without_kg():
  response = client.post("/chat/send", json={"query": "hello"})
  assert response.status_code == 404


def test_send():
  kg_mock = Mock(spec=KnowledgeGraph)
  kg_mock.get_file_tree.return_value = "mock_file_tree"
  neo4j_driver_mock = Mock(spec=neo4j.Driver)
  mock_response = "Mock response"
  fake_llm = FakeListChatModelWithTools(responses=[mock_response])
  fake_agent = ContextProviderAgent(
    llm=fake_llm, kg=kg_mock, neo4j_driver=neo4j_driver_mock
  )

  app.state.kg = kg_mock
  app.state.neo4j_driver = neo4j_driver_mock
  app.state.cp_agent = fake_agent

  response = client.post("/chat/send", json={"query": "hello"})
  assert response.status_code == 200
  assert response.json() == mock_response
