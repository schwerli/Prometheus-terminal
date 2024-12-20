import logging
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.utils.issue_util import format_issue_info
from prometheus.utils.lang_graph_util import extract_ai_responses, extract_human_queries


class ContextRefineStructuredOutput(BaseModel):
  reasoning: str = Field(description="Your step by step reasoning.")
  refined_query: str = Field(
    "Additional query to ask the ContextRetriever if the context is not enough. Empty otherwise."
  )


class ContextRefineNode:
  SYS_PROMPT = """\
You are a software engineering assistant specialized in analyzing code context to determine if
additional source code or documentation from the codebase is necessary.

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

Analyze if the current context is sufficient to fix the issue. The context is sufficient if a persion
without prior knowledge about the codebase can understand the issue and fix it. Do not ask the same
query as before.

For example you can ask for additional files in the codebase to be looked at, search method/class
implementations, or ask more for context in the same file.
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
    bug_context = "\n\n".join(extract_ai_responses(state["context_provider_messages"]))
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
