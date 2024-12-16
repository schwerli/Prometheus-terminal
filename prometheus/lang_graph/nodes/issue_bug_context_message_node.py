import logging
from typing import Dict

from langchain_core.messages import HumanMessage

from prometheus.utils.issue_util import format_issue_info


class IssueBugContextMessageNode:
  HUMAN_PROMPT = """\
{issue_info}

OBJECTIVE: Retrieve COMPLETE source code context needed to understand and fix this bug, focusing on actual implementation files while excluding test files. The retrieved context must be comprehensive enough for someone with zero knowledge of the codebase to fully understand and fix this bug.

<retrieval_priority>
1. HIGH PRIORITY:
   - Files explicitly mentioned in the issue description or comments
   - Classes/functions directly referenced in any code examples
   - Files in the same directory as mentioned files

2. SECONDARY PRIORITY:
   - Dependencies of high-priority files
   - Related configuration files
   - Utility functions used by high-priority code
</retrieval_priority>

<retrieval_rules>
1. INCLUDE:
   - Complete class definitions with ALL methods
   - Parent/super classes and their complete implementations
   - Imported dependencies and their implementations
   - Configuration files affecting behavior
   - Utility functions used by any relevant code

2. EXCLUDE:
   - Test files (ending in _test.py, test_.py, or within test/ directories)
   - Example code files
   - Documentation files
   - Backup files
</retrieval_rules>

<retrieval_strategy>
1. Start with files/classes/functions explicitly mentioned in the issue
2. Trace UPWARD: Find parent classes, interfaces, abstractions
3. Trace DOWNWARD: Find child classes, implementations
4. Trace SIDEWAYS: Find imported dependencies, utility functions
5. Trace CONFIGURATION: Find all config values affecting behavior
</retrieval_strategy>

<example>
Given bug report:
Issue title: calculate_area returns wrong result
Issue description:
The bug seems to be in geometry/shapes.py. Here's the reproduction:
```python
rectangle = Rectangle(10, 5)
result = rectangle.calculate_area()
assert result == 50  # AssertionError: actual value is 75
```

Retrieval Process:
<step1>
<thought>First check the explicitly mentioned file geometry/shapes.py, then find the complete Rectangle class implementation</thought>
<action>find geometry/shapes.py and Rectangle class implementation NOT in test files
# File: geometry/shapes.py
[Complete file contents...]

# File: geometry/rectangle.py
[Complete Rectangle class implementation...]
</action>
</step1>

<step2>
<thought>Find complete parent class implementation</thought>
<action>find Figure class implementation NOT in test files
# File: geometry/base.py
[Complete Figure class implementation...]
</action>
</step2>

<step3>
<thought>Find all imported dependencies and their implementations</thought>
<action>find implementations of validate_dimensions and config values NOT in test files
# File: geometry/utils.py
[validate_dimensions implementation...]

# File: geometry/config.py
[Configuration values...]
</action>
</step3>

<step4>
<thought>Verify we have complete context by checking all code paths</thought>
<action>Return consolidated context from all found files NOT including tests
[Combined complete context...]
</action>
</step4>
</example>

CRITICAL REQUIREMENTS:
1. ALWAYS start with files explicitly mentioned in the issue
2. Retrieve COMPLETE class/function implementations, not just relevant methods
3. NEVER include test files or test-related code
4. Follow ALL possible execution paths
5. Find explicit source code evidence for everything - make no assumptions
6. Continue until ALL relevant implementation code is found
7. Include ALL configuration values that could affect behavior
"""

  def __init__(self):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.issue_bug_context_message_node")

  def __call__(self, state: Dict):
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(
        issue_info=format_issue_info(
          state["issue_title"], state["issue_body"], state["issue_comments"]
        ),
      )
    )
    self._logger.debug(f"Sending query to ContextProviderNode:\n{human_message}")
    return {"context_provider_messages": [human_message]}
