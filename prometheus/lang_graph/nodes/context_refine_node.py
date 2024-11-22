import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.context_provider_state import ContextProviderState


class ContextRefineOutput(BaseModel):
  has_sufficient_context: bool = Field(
    description="If the context is sufficient for someone with no prior knowledge about the codebase can address the query."
  )
  refined_query: str = Field(
    description="Query to ask the ContextProvider to guide it to find more relevant context."
  )


class ContextRefineNode:
  SYS_PROMPT = """\
You are a Context Refinement Agent that evaluates whether the provided context is sufficient to answer
a query and guides the ContextProvider to find more relevant context if needed.

Your task:
1. Determine if the current context is sufficient for someone with no prior knowledge about the codebase to address the query
2. If context is insufficient, generate a refined query to help ContextProvider find more relevant information.
   The refined query should be self-contained.

Example 1:
Input:
```
Original query: How does the error handling work in the authentication system?

ContextProvider previous responses:
The authentication system uses JWT tokens for user verification.

ContextProvider current response:
The auth.py file contains basic token validation logic.
```

Output:
```json
{
  "has_sufficient_context": false,
  "refined_query": "Show me the error handling code and exception types in auth.py, particularly around token validation and user authentication failures"
}
```

Example 2:
Input:
```
Original query: What's the database schema for users?

ContextProvider previous responses:
The User model has fields for username, email, and password.
Table includes created_at and updated_at timestamps.

ContextProvider current response:
Found database migration showing User table with foreign keys to roles and preferences tables.
```

Output:
```json
{
  "has_sufficient_context": true,
  "refined_query": ""
}
```

DO NOT ASK QUESTION THAT ContextProvider PREVIOUSLY HAS RESPONDED.
""".replace("{", "{{").replace("}", "}}")

  HUMAN_PROMPT = """\
Original query:
{original_query}

ContextProvider preivous responses:
{previous_responses}

ContextProvider current response:
{current_response}
"""

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{all_context_info}")]
    )
    structured_llm = model.with_structured_output(ContextRefineOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.context_refine_node")

  def format_human_message(self, state: ContextProviderState) -> str:
    previous_responses = ""
    if "all_context_provider_responses" in state and state["all_context_provider_responses"]:
      for response in state["all_context_provider_responses"]:
        previous_responses += response.content + "\n"

    return self.HUMAN_PROMPT.format(
      original_query=state["original_query"],
      previous_responses=previous_responses,
      current_response=state["context_provider_messages"][-1].content,
    )

  def __call__(self, state: ContextProviderState):
    all_context_info = self.format_human_message(state)

    response = self.model.invoke({"all_context_info": all_context_info})
    self._logger.debug(f"ContextRefineOutput response:\n{response}")

    return {
      "has_sufficient_context": response.has_sufficient_context,
      "refined_query": response.refined_query,
      "all_context_provider_responses": state["context_provider_messages"][-1].content,
    }
