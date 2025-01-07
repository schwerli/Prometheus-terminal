import logging
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.utils.issue_util import format_agent_tool_message_history, format_issue_info
from prometheus.utils.lang_graph_util import extract_human_queries


class ContextRefineStructuredOutput(BaseModel):
  reasoning: str = Field(description="Your step by step reasoning.")
  refined_query: str = Field(
    "Additional query to ask the ContextRetriever if the context is not enough. Empty otherwise."
  )


class ContextRefineNode:
  SYS_PROMPT = """\
You are a software engineering assistant specialized in analyzing code context to determine if
additional source code or documentation from the codebase is necessary.

Your goal is to request additional context ONLY when necessary:
1. When critical implementation details are missing to understand the root cause
2. When key dependencies or related code are not visible in the current context
3. When documentation is needed to understand complex business logic or requirements

DO NOT request additional context if:
1. The current context already contains enough information to implement a fix
2. The additional context would only provide nice-to-have but non-essential details

Output format:
```python
class ContextRefineStructuredOutput(BaseModel):
    reasoning: str     # Why current context is/isn't enough
    refined_query: str # Additional query to ask the ContextRetriever if the context is not enough. Empty otherwise
```

The codebase structure:
{file_tree}
"""

  REFINE_PROMPT = """\
I have the following issue:
{issue_info}

Here is the queries I have asked:
{queries}

All aggregated context for the queries:
{bug_context}

Analyze if the current context is sufficient to implement a fix by considering:
1. Can you identify the root cause of the issue from the current context?
2. Do you have access to all relevant code that needs to be modified?
3. Are all critical dependencies and their interfaces visible?

Only request additional context if essential information is missing. Ensure you're not requesting:
- Information already provided in previous queries
- Nice-to-have but non-essential details

If additional context is needed, be specific about what you're looking for.
"""

  def __init__(self, model: BaseChatModel, kg: KnowledgeGraph):
    prompt = ChatPromptTemplate.from_messages(
      [
        ("system", self.SYS_PROMPT.format(file_tree=kg.get_file_tree())),
        ("human", "{human_prompt}"),
      ]
    )
    structured_llm = model.with_structured_output(ContextRefineStructuredOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.context_refine_node")

  def format_refine_message(self, state: Dict):
    bug_context = format_agent_tool_message_history(state["context_provider_messages"])
    queries = "\n\n".join(extract_human_queries(state["context_provider_messages"])[1:])
    return self.REFINE_PROMPT.format(
      issue_info=format_issue_info(
        state["issue_title"], state["issue_body"], state["issue_comments"]
      ),
      queries=queries,
      bug_context=bug_context,
    )

  def __call__(self, state: Dict):
    if "max_refined_query_loop" in state and state["max_refined_query_loop"] == 0:
      self._logger.info("Reached max_refined_query_loop, not asking for more context")
      return {"refined_query": ""}

    human_prompt = self.format_refine_message(state)
    self._logger.debug(human_prompt)
    response = self.model.invoke({"human_prompt": human_prompt})
    self._logger.debug(response)

    state_update = {"refined_query": response.refined_query}

    if "max_refined_query_loop" in state:
      state_update["max_refined_query_loop"] = state["max_refined_query_loop"] - 1

    if response.refined_query:
      state_update["context_provider_messages"] = [HumanMessage(content=response.refined_query)]

    return state_update
