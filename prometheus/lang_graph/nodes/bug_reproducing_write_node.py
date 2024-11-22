import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState


class BugReproducingWriteStructuredOutput(BaseModel):
  bug_reproducing_code: str = Field(description="The self-contained code that reproduce the bug")


class BugReproducingWriteNode:
  SYS_PROMPT = """\
You are an agent that generates minimal self-contained code that reproduce reported bugs.
Your generated code must correctly identify and report the bug status.

REQUIREMENTS:
1. Create minimal, self-contained, complete, runnable code that checks for the bug
2. ALWAYS print exactly "Bug reproduced" when the bug is found
3. ALWAYS print exactly "Bug resolved" when the bug is fixed
4. Include all necessary imports
5. Add clear comments explaining what the bug is

EXAMPLES:

Python Example (String Case Bug):
```python
def check_case_sensitive_replace():
    # Bug: str.replace() fails to respect case sensitivity
    text = "Hello HELLO hello"
    
    try:
        result = text.replace("hello", "hi", case_sensitive=True)  # Bug: parameter doesn't exist
        print("Bug resolved")
    except TypeError:
        print("Bug reproduced")

if __name__ == "__main__":
    check_case_sensitive_replace()
```

JavaScript Example (Array Sum Bug):
```javascript
function checkNegativeArraySum() {
    // Bug: Array.reduce() gives wrong results with negative numbers
    const numbers = [-1, -2, -3];
    const sum = numbers.reduce((a, b) => a + b, -0);  // Bug: -0 causes incorrect sign
    
    if (sum === -6) {
        console.log("Bug resolved");
    } else {
        console.log("Bug reproduced");
    }
}

checkNegativeArraySum();
```

RESPONSE FORMAT:
1. Write minimal but complete code
2. Check for bug presence
3. ALWAYS use exact messages:
   - Print "Bug reproduced" when bug is found
   - Print "Bug resolved" when bug is fixed

Remember: Focus on clearly identifying the bug state with the correct output message!
""".replace("{", "{{").replace("}", "}}")

  HUMAN_PROMPT = """\
ISSUE INFORMATION:
Title: {title}
Description: {body}
Comments: {comments}

Bug context summary:
{bug_context}

Previous bug reproducing code
{previous_bug_reproducing_code}

Previous bug reproducing fail log
{previous_bug_reproducing_fail_log}
"""

  def __init__(self, model: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
      [("system", self.SYS_PROMPT), ("human", "{issue_info}")]
    )
    structured_llm = model.with_structured_output(BugReproducingWriteStructuredOutput)
    self.model = prompt | structured_llm
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.bug_reproducing_write_node")

  def format_human_message(self, state: BugReproductionState):
    previous_bug_reproducing_code = ""
    if "bug_reproducing_code" in state and state["bug_reproducing_code"]:
      previous_bug_reproducing_code = state["bug_reproducing_code"]
    previous_bug_reproducing_fail_log = ""
    if "reproduced_bug_failure_log" in state and state["reproduced_bug_failure_log"]:
      previous_bug_reproducing_fail_log = state["reproduced_bug_failure_log"]
    return self.HUMAN_PROMPT.format(
      title=state["issue_title"],
      body=state["issue_body"],
      comments=state["issue_comments"],
      bug_context=state["bug_context"],
      previous_bug_reproducing_code=previous_bug_reproducing_code,
      previous_bug_reproducing_fail_log=previous_bug_reproducing_fail_log,
    )

  def __call__(self, state: BugReproductionState):
    issue_info = self.format_human_message(state)
    response = self.model.invoke({"issue_info": issue_info})

    self._logger.debug(f"BugReproducingWriteNode response:\n{response}")
    return {"bug_reproducing_code": response.bug_reproducing_code}
