import logging

from langchain_core.messages import HumanMessage

from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from prometheus.utils.issue_util import format_issue_info


class IssueBugContextMessageNode:
  HUMAN_PROMPT = """\
{issue_info}

OBJECTIVE: Trace execution path and gather all code needed to understand and fix this bug.

<analysis>
To identify the root cause, we need to:

1. Trace execution path:
   - Find where execution starts
   - Follow the call chain
   - Identify state modifications

2. Find definition of involved code:
   - Find all relevant classes/methods
   - Find parent classes and interfaces
   - Find helper functions used

3. Analyze potential causes:
   - Check initialization logic
   - Verify calculations
   - Look for state corruption
   - Check inheritance behavior
</analysis>

<example>
A user has reported the following issue to the codebase:
Issue title: calculate_area returns wrong result

Issue description:
When calculating the area of a rectangle, it returns the wrong result. Below is a minimal reproducing example:

```python
rectangle = Rectangle(10, 5)
result = rectangle.calculate_area()
assert result == 50  # AssertionError: actual value is 75
```

Analysis Steps:
1. First, we need Rectangle class definition and its parent class
2. Then check Rectangle initialization and any methods called during it
3. Finally examine calculate_area implementation and its dependencies

Required Context:
# File: geometry/base.py
from abc import ABC, abstractmethod

class Figure(ABC):
    def __init__(self):
        self._cached_area = None
        self._scale_factor = 1.0
    
    @abstractmethod
    def calculate_area(self):
        pass
    
    def set_scale(self, factor):
        self._scale_factor = factor
        self._cached_area = None

# File: geometry/rectangle.py
from .base import Figure
from .utils import validate_dimensions
from .config import ENABLE_CACHING

class Rectangle(Figure):
    def __init__(self, length, width):
        super().__init__()
        validate_dimensions(length, width)
        self.length = length
        self.width = width
    
    def calculate_area(self):
        if ENABLE_CACHING and self._cached_area is not None:
            return self._cached_area
            
        area = self.length * self.width * self._scale_factor
        
        if ENABLE_CACHING:
            self._cached_area = area
            
        return area

# File: geometry/utils.py
def validate_dimensions(*args):
    for dim in args:
        if not isinstance(dim, (int, float)):
            raise TypeError(f"Dimension must be numeric, got {{type(dim)}}")
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {{dim}}")

# File: geometry/config.py
ENABLE_CACHING = True
DEFAULT_SCALE = 1.5  # This could be the issue - default scale factor is 1.5

Potential Issues:
1. Parent class Figure initializes _scale_factor = 1.0
2. config.py sets DEFAULT_SCALE = 1.5
3. Area caching might retain stale values
4. Scale factor affects all calculations

Additional Context Needed:
1. Where is DEFAULT_SCALE being applied?
2. Are there other places that modify _scale_factor?
3. Check for dimension validation side effects
4. Review caching behavior impact
</example>

CONSTRAINTS:
- Include complete file paths
- Show full implementations
- Preserve all comments
- Include relevant line numbers
- Show configuration context

Using this issue description, trace through the code to gather ALL relevant context needed to understand and fix the bug.
"""

  def __init__(self):
    self._logger = logging.getLogger("prometheus.lang_graph.nodes.issue_bug_context_message_node")

  def __call__(self, state: BugReproductionState):
    human_message = HumanMessage(
      self.HUMAN_PROMPT.format(
        issue_info=format_issue_info(
          state["issue_title"], state["issue_body"], state["issue_comments"]
        ),
      )
    )
    self._logger.debug(f"Sending query to ContextProviderNode:\n{human_message}")
    return {"context_provider_messages": [human_message]}
