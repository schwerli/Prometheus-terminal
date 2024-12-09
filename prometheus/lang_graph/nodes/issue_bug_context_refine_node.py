import logging
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.issue_bug_state import IssueBugState
from prometheus.utils.issue_util import format_issue_info
from prometheus.utils.lang_graph_util import extract_ai_responses


class IssueBugContextRefineStructuredOutput(BaseModel):
  reasoning: str = Field(description="Your step by step reasoning.")
  refined_query: str = Field(
    "Additional query to ask the ContextRetriever if the context is not enough. Empty otherwise."
  )


class IssueBugContextRefineNode:
  SYS_PROMPT = """\
You are a software engineering assistant specialized in analyzing build and test failures to determine if
additional context from the codebase is needed. Your task is to analyze errors and determine if fetching
more code context could help resolve the issue.

Step-by-Step Analysis Process:
1. Examine the current error and available context
2. Identify what parts of the code are involved in the error
3. Check if you have all relevant code including:
   - Complete implementations
   - Parent classes/interfaces 
   - Related configuration files
4. Determine if accessing more code context could help resolve the error
5. If needed, formulate a specific query to get more code context

<examples>
<example>
Input:
Issue: "Authentication fails in CustomAuthProvider"
Context: "CustomAuthProvider.java implementation"
Patch: "Updated authentication logic"
Error: "Build failed: Cannot resolve symbol 'BaseAuthProvider'"

Thought process:
1. Error indicates a missing required class
2. Current context only shows CustomAuthProvider
3. BaseAuthProvider code is not included but needed
4. Getting this parent class would help resolve the error

Output:
{
  "reasoning": "The build error shows CustomAuthProvider depends on BaseAuthProvider, but we don't have BaseAuthProvider's code. Getting this code would help understand the required interface/implementation.",
  "refined_query": "Find the complete implementation of BaseAuthProvider class that CustomAuthProvider extends/implements"
}
</example>

<example>
Input:
Issue: "NullPointerException in UserService"
Context: "Full UserService.java implementation"
Patch: "Added null checks"
Error: "Test failure: NullPointerException in UserService.validateUser()"

Thought process:
1. Have full UserService implementation
2. Error occurs in code we can already see
3. Getting more code won't help with this NPE
4. This is a logic error in the existing code

Output:
{
  "reasoning": "We already have the complete UserService implementation including the validateUser method. The NPE is occurring within code we can already see. This appears to be a logic error in the null checking, not a missing context issue.",
  "refined_query": ""
}
</example>
</examples>

Your output must strictly follow this Pydantic model:
```python
class IssueBugContextRefineStructuredOutput(BaseModel):
    reasoning: str     # Your step by step reasoning about the context needs
    refined_query: str # Additional query if context is insufficient, empty string otherwise
```

Important Guidelines:
1. Only request additional context when clearly needed to resolve the error
2. Make refined queries specific to the code/files mentioned in the error
3. Return empty refined_query if:
   - Current context is sufficient
   - The error is a logic/implementation issue
4. Focus queries on related code that would help understand/fix the error
5. Be specific about what code you need and why it would help resolve the error
""".replace("{", "{{").replace("}", "}}")

  HUMAN_PROMPT = """\
{issue_info}

This was the context provided to the edit agent:
{bug_context}

The edit agent generated the following patch:
{edit_patch}

The patch generated following error:
{edit_error}

Now analyze if the error was caused by not having enough context.
"""

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{edit_context}")]
    )
    structured_llm = model.with_structured_output(IssueBugContextRefineStructuredOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger(
      "prometheus.lang_graph.nodes.issue_bug_context_refine_message_node"
    )

  def format_human_message(self, state: IssueBugState):
    edit_error = ""
    if "reproducing_test_fail_log" in state and state["reproducing_test_fail_log"]:
      edit_error = (
        f"Your failed to pass the bug exposing test cases:\n{state['reproducing_test_fail_log']}"
      )
    elif "build_fail_log" in state and state["build_fail_log"]:
      edit_error = f"Your failed to pass the build:\n{state['build_fail_log']}"
    elif "existing_test_fail_log" in state and state["existing_test_fail_log"]:
      edit_error = f"Your failed to existing test cases:\n{state['existing_test_fail_log']}"

    assert edit_error != ""

    bug_context = "\n".join(extract_ai_responses(state["context_provider_messages"]))

    return self.HUMAN_PROMPT.format(
      issue_info=format_issue_info(
        state["issue_title"], state["issue_body"], state["issue_comments"]
      ),
      bug_context=bug_context,
      edit_patch=state["edit_patch"],
      edit_error=edit_error,
    )

  def __call__(self, state: Dict):
    edit_context = self.format_human_message(state)
    response = self.model.invoke({"edit_context": edit_context})
    self._logger.debug(f"IssueBugContextRefineNode response:\n{response}")

    state_update = {"refined_query": response.refined_query}
    if response.refined_query:
      state_update["context_provider_messages"] = [HumanMessage(content=response.refined_query)]

    return state_update
